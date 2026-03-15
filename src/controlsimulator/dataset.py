from __future__ import annotations

import hashlib
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import repeat
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from controlsimulator.config import DatasetConfig, save_config_snapshot
from controlsimulator.metrics import extract_response_metrics
from controlsimulator.plants import Plant, heuristic_pid_scales, sample_pid_gains, sample_plant
from controlsimulator.plotting import (
    plot_dataset_family_stability,
    plot_gain_distributions,
    plot_trajectory_amplitudes,
)
from controlsimulator.simulate import closed_loop_is_stable, simulate_closed_loop
from controlsimulator.splits import assert_no_plant_leakage, assign_dataset_splits
from controlsimulator.utils import dump_json, ensure_dir, load_json, resolve_path

MANIFEST_FILENAME = "manifest.json"
SAMPLES_FILENAME = "samples.parquet"
TRAJECTORIES_FILENAME = "trajectories.npz"


@dataclass(slots=True)
class DatasetBundle:
    samples: pd.DataFrame
    trajectories: np.ndarray
    time_grid: np.ndarray
    metadata: dict[str, Any]
    dataset_dir: Path


def time_grid_from_config(config: DatasetConfig) -> np.ndarray:
    return np.linspace(0.0, config.t_final, config.n_time_steps, dtype=np.float32)


def generate_dataset(config: DatasetConfig) -> Path:
    dataset_dir = ensure_dir(resolve_path(config.dataset_dir()))
    chunks_dir = ensure_dir(dataset_dir / "chunks")
    manifest = _build_manifest(config)
    _validate_or_initialize_dataset_dir(dataset_dir, chunks_dir, manifest)
    save_config_snapshot(config, dataset_dir / "config_snapshot.yaml")

    time_grid = time_grid_from_config(config)
    np.save(dataset_dir / "time_grid.npy", time_grid)

    total_chunks = int(manifest["total_chunks"])
    generation_start = perf_counter()
    executor: ProcessPoolExecutor | None = None
    if config.num_workers > 1:
        executor = ProcessPoolExecutor(max_workers=config.num_workers)

    try:
        for chunk_index in tqdm(range(total_chunks), desc=f"dataset:{config.name}"):
            metadata_path = _chunk_metadata_path(chunks_dir, chunk_index)
            trajectory_path = _chunk_trajectory_path(chunks_dir, chunk_index)
            if metadata_path.exists() and trajectory_path.exists():
                continue
            start_plant = chunk_index * config.chunk_size_plants
            end_plant = min(config.n_plants, start_plant + config.chunk_size_plants)
            frame, trajectories = _generate_chunk(
                config=config,
                time_grid=time_grid,
                start_plant=start_plant,
                end_plant=end_plant,
                executor=executor,
            )
            _validate_chunk_health(frame)
            frame.to_parquet(metadata_path, index=False, compression="zstd")
            np.savez_compressed(trajectory_path, trajectories=trajectories.astype(np.float32))
    finally:
        if executor is not None:
            executor.shutdown()

    dataset_dir = consolidate_dataset(config)
    metadata = load_json(dataset_dir / "metadata.json")
    metadata["generation_seconds"] = perf_counter() - generation_start
    metadata["dataset_size_bytes"] = _directory_size_bytes(dataset_dir)
    dump_json(metadata, dataset_dir / "metadata.json")
    return dataset_dir


def consolidate_dataset(config: DatasetConfig) -> Path:
    dataset_dir = ensure_dir(resolve_path(config.dataset_dir()))
    chunks_dir = dataset_dir / "chunks"
    manifest_path = dataset_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest = load_json(manifest_path)

    metadata_paths = sorted(chunks_dir.glob("metadata_*.parquet"))
    trajectory_paths = sorted(chunks_dir.glob("trajectories_*.npz"))
    expected_indices = set(range(int(manifest["total_chunks"])))
    actual_metadata_indices = {_chunk_index_from_path(path) for path in metadata_paths}
    actual_trajectory_indices = {_chunk_index_from_path(path) for path in trajectory_paths}
    if actual_metadata_indices != expected_indices or actual_trajectory_indices != expected_indices:
        raise RuntimeError(
            "Chunk files do not match the manifest. "
            "Regenerate the missing chunk files or use a new dataset name."
        )

    samples = pd.concat([pd.read_parquet(path) for path in metadata_paths], ignore_index=True)
    trajectories = np.concatenate(
        [np.load(path)["trajectories"] for path in trajectory_paths],
        axis=0,
    ).astype(np.float32)
    if samples.shape[0] != trajectories.shape[0]:
        raise RuntimeError("Metadata row count does not match trajectory row count.")

    samples["split"] = assign_dataset_splits(
        samples,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        ood_families=config.ood_families,
        seed=config.seed,
    )
    assert_no_plant_leakage(samples)

    _validate_dataset_health(samples, trajectories, config)

    samples.to_parquet(dataset_dir / SAMPLES_FILENAME, index=False, compression="zstd")
    np.savez_compressed(dataset_dir / TRAJECTORIES_FILENAME, trajectories=trajectories)

    metadata = _dataset_metadata(config, samples, trajectories)
    dump_json(metadata, dataset_dir / "metadata.json")
    _write_dataset_diagnostic_plots(config, dataset_dir, samples, trajectories)
    return dataset_dir


def load_dataset(dataset_dir: str | Path) -> DatasetBundle:
    resolved = resolve_path(dataset_dir)
    samples_path = resolved / SAMPLES_FILENAME
    if samples_path.exists():
        samples = pd.read_parquet(samples_path)
    else:
        samples = pd.read_csv(resolved / "samples.csv.gz")

    trajectories_path = resolved / TRAJECTORIES_FILENAME
    if trajectories_path.exists():
        trajectories = np.load(trajectories_path)["trajectories"]
    else:
        trajectories = np.load(resolved / "trajectories.npy")

    time_grid = np.load(resolved / "time_grid.npy")
    metadata = load_json(resolved / "metadata.json")
    return DatasetBundle(
        samples=samples,
        trajectories=trajectories,
        time_grid=time_grid,
        metadata=metadata,
        dataset_dir=resolved,
    )


def _generate_chunk(
    config: DatasetConfig,
    time_grid: np.ndarray,
    start_plant: int,
    end_plant: int,
    executor: ProcessPoolExecutor | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    plant_ids = list(range(start_plant, end_plant))
    if executor is None:
        plant_batches = [
            _generate_single_plant(config, time_grid, plant_id) for plant_id in plant_ids
        ]
    else:
        chunksize = max(1, len(plant_ids) // max(config.num_workers * 4, 1))
        plant_batches = list(
            executor.map(
                _generate_single_plant,
                repeat(config),
                repeat(time_grid),
                plant_ids,
                chunksize=chunksize,
            )
        )

    rows: list[dict[str, Any]] = []
    trajectories: list[np.ndarray] = []
    for plant_rows, plant_trajectories in plant_batches:
        rows.extend(plant_rows)
        trajectories.extend(plant_trajectories)

    return pd.DataFrame(rows), np.vstack(trajectories)


def _generate_single_plant(
    config: DatasetConfig,
    time_grid: np.ndarray,
    plant_id: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rng = np.random.default_rng(config.seed + (plant_id * 9_973))
    plant = sample_plant(rng, plant_id=plant_id, families=config.families)
    rows: list[dict[str, Any]] = []
    trajectories: list[np.ndarray] = []
    sampled_gains: list[dict[str, Any]] = []

    wide_count = int(round(config.controllers_per_plant * config.wide_sampling_fraction))
    wide_count = min(max(wide_count, 1), config.controllers_per_plant)
    targeted_count = config.controllers_per_plant - wide_count

    for controller_index in range(config.controllers_per_plant):
        if controller_index < wide_count:
            kp, ki, kd = sample_pid_gains(
                rng,
                plant,
                tuple(config.kp_multiplier_range),
                tuple(config.ki_multiplier_range),
                tuple(config.kd_multiplier_range),
            )
            sampling_mode = "wide_random"
        else:
            kp, ki, kd, sampling_mode = _sample_targeted_pid_gains(
                rng,
                plant,
                config,
                sampled_gains,
            )

        result = simulate_closed_loop(
            plant=plant,
            kp=kp,
            ki=ki,
            kd=kd,
            tau_d=config.derivative_filter_tau,
            time_grid=time_grid,
        )
        is_usable, failure_reason, trajectory_peak_abs, final_value = _apply_result_safety_filters(
            result,
            config,
        )

        sample_id = (plant_id * config.controllers_per_plant) + controller_index
        row = {
            "sample_id": sample_id,
            "plant_id": plant_id,
            "controller_index": controller_index,
            "plant_family": plant.family,
            "plant_order": plant.plant_order,
            "dc_gain": plant.dc_gain,
            "dominant_pole_mag": plant.dominant_pole_mag,
            "mean_pole_mag": plant.mean_pole_mag,
            "dominant_time_constant": 1.0 / plant.dominant_pole_mag,
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "tau_d": config.derivative_filter_tau,
            "root_stable": bool(result.stability_margin < -1e-6),
            "stable": bool(is_usable),
            "stability_margin": result.stability_margin,
            "failure_reason": failure_reason,
            "gain_sampling_mode": sampling_mode,
            "trajectory_peak_abs": trajectory_peak_abs,
            "trajectory_final_value": final_value,
        }

        padded_num = plant.padded_numerator()
        padded_den = plant.padded_denominator()
        for index, value in enumerate(padded_num):
            row[f"num_{index}"] = float(value)
        for index, value in enumerate(padded_den):
            row[f"den_{index}"] = float(value)

        if is_usable and result.trajectory is not None:
            response_metrics = extract_response_metrics(
                time_grid,
                result.trajectory,
                peak_control_effort=result.peak_control_effort,
            )
            row.update(response_metrics.to_dict())
            trajectories.append(result.trajectory.astype(np.float32))
        else:
            row.update(
                {
                    "overshoot_pct": np.nan,
                    "rise_time": np.nan,
                    "settling_time": np.nan,
                    "steady_state_error": np.nan,
                    "peak_control_effort": np.nan,
                }
            )
            trajectories.append(np.full_like(time_grid, np.nan, dtype=np.float32))

        rows.append(row)
        sampled_gains.append(
            {
                "kp": kp,
                "ki": ki,
                "kd": kd,
                "stable": bool(is_usable),
            }
        )

    if targeted_count <= 0:
        return rows, trajectories
    return rows, trajectories


def _sample_targeted_pid_gains(
    rng: np.random.Generator,
    plant: Plant,
    config: DatasetConfig,
    sampled_gains: list[dict[str, Any]],
) -> tuple[float, float, float, str]:
    low_bounds, high_bounds, base_scales = _gain_bounds(plant, config)

    stable_anchor = _find_stable_anchor(rng, plant, config, sampled_gains, base_scales, low_bounds)
    unstable_anchor = _find_unstable_anchor(
        rng,
        plant,
        config,
        sampled_gains,
        stable_anchor,
        low_bounds,
        high_bounds,
    )
    if unstable_anchor is None:
        kp, ki, kd = sample_pid_gains(
            rng,
            plant,
            tuple(config.kp_multiplier_range),
            tuple(config.ki_multiplier_range),
            tuple(config.kd_multiplier_range),
        )
        return kp, ki, kd, "targeted_fallback_random"

    stable_log = np.log(np.clip(stable_anchor, low_bounds, high_bounds))
    unstable_log = np.log(np.clip(unstable_anchor, low_bounds, high_bounds))
    for _ in range(config.boundary_search_steps):
        mid_log = 0.5 * (stable_log + unstable_log)
        mid_gains = np.exp(mid_log)
        if closed_loop_is_stable(
            plant,
            float(mid_gains[0]),
            float(mid_gains[1]),
            float(mid_gains[2]),
            config.derivative_filter_tau,
        ):
            stable_log = mid_log
        else:
            unstable_log = mid_log

    alpha = float(np.clip(0.5 + rng.normal(0.0, config.boundary_mix_std), 0.15, 0.85))
    sample_log = ((1.0 - alpha) * stable_log) + (alpha * unstable_log)
    sample_log += rng.normal(0.0, config.boundary_jitter_std, size=3)
    candidate = np.exp(sample_log)
    candidate = np.clip(candidate, low_bounds, high_bounds)
    return float(candidate[0]), float(candidate[1]), float(candidate[2]), "boundary_search"


def _find_stable_anchor(
    rng: np.random.Generator,
    plant: Plant,
    config: DatasetConfig,
    sampled_gains: list[dict[str, Any]],
    base_scales: np.ndarray,
    low_bounds: np.ndarray,
) -> np.ndarray:
    stable_records = [record for record in sampled_gains if record["stable"]]
    if stable_records:
        record = stable_records[int(rng.integers(0, len(stable_records)))]
        return np.asarray([record["kp"], record["ki"], record["kd"]], dtype=np.float64)

    candidate = np.clip(base_scales, low_bounds, None)
    if closed_loop_is_stable(
        plant,
        float(candidate[0]),
        float(candidate[1]),
        float(candidate[2]),
        config.derivative_filter_tau,
    ):
        return candidate

    for scale in [0.7, 0.5, 0.35, 0.2, 0.1, 0.05]:
        candidate = np.clip(base_scales * scale, low_bounds, None)
        if closed_loop_is_stable(
            plant,
            float(candidate[0]),
            float(candidate[1]),
            float(candidate[2]),
            config.derivative_filter_tau,
        ):
            return candidate
    return low_bounds


def _find_unstable_anchor(
    rng: np.random.Generator,
    plant: Plant,
    config: DatasetConfig,
    sampled_gains: list[dict[str, Any]],
    stable_anchor: np.ndarray,
    low_bounds: np.ndarray,
    high_bounds: np.ndarray,
) -> np.ndarray | None:
    unstable_records = [record for record in sampled_gains if not record["stable"]]
    if unstable_records:
        record = unstable_records[int(rng.integers(0, len(unstable_records)))]
        return np.asarray([record["kp"], record["ki"], record["kd"]], dtype=np.float64)

    stable_records = [record for record in sampled_gains if record["stable"]]
    candidate_sources = stable_records or [
        {"kp": stable_anchor[0], "ki": stable_anchor[1], "kd": stable_anchor[2]}
    ]
    for _ in range(12):
        source = candidate_sources[int(rng.integers(0, len(candidate_sources)))]
        source_gains = np.asarray([source["kp"], source["ki"], source["kd"]], dtype=np.float64)
        amplification = np.exp(np.abs(rng.normal(loc=0.8, scale=0.45, size=3)))
        candidate = np.clip(source_gains * amplification, low_bounds, high_bounds)
        if not closed_loop_is_stable(
            plant,
            float(candidate[0]),
            float(candidate[1]),
            float(candidate[2]),
            config.derivative_filter_tau,
        ):
            return candidate

    for _ in range(8):
        top_heavy = np.exp(rng.uniform(np.log(high_bounds / 4.0), np.log(high_bounds)))
        candidate = np.clip(top_heavy, low_bounds, high_bounds)
        if not closed_loop_is_stable(
            plant,
            float(candidate[0]),
            float(candidate[1]),
            float(candidate[2]),
            config.derivative_filter_tau,
        ):
            return candidate
    return None


def _apply_result_safety_filters(
    result: Any,
    config: DatasetConfig,
) -> tuple[bool, str, float, float]:
    if not result.stable or result.trajectory is None:
        return False, result.reason or "unstable_closed_loop", float("nan"), float("nan")

    trajectory_peak_abs = float(np.max(np.abs(result.trajectory)))
    final_value = float(result.trajectory[-1])
    if trajectory_peak_abs > config.max_abs_trajectory:
        return False, "trajectory_limit_exceeded", trajectory_peak_abs, final_value
    if (result.peak_control_effort or 0.0) > config.max_peak_control_effort:
        return False, "control_effort_limit_exceeded", trajectory_peak_abs, final_value
    return True, "", trajectory_peak_abs, final_value


def _validate_chunk_health(frame: pd.DataFrame) -> None:
    unstable_fraction = 1.0 - float(frame["stable"].mean())
    if unstable_fraction > 0.5:
        raise RuntimeError(
            f"Chunk unstable fraction exceeded safety threshold: {unstable_fraction:.3f}"
        )

    forbidden_reasons = {
        "non_finite_response",
        "trajectory_limit_exceeded",
    }
    reason_counts = (
        frame.loc[frame["failure_reason"] != "", "failure_reason"].value_counts().to_dict()
    )
    unexpected = forbidden_reasons.intersection(reason_counts)
    if unexpected:
        raise RuntimeError(
            "Dataset generation aborted due to unsafe trajectories or non-finite outputs: "
            f"{sorted(unexpected)}"
        )


def _validate_dataset_health(
    samples: pd.DataFrame,
    trajectories: np.ndarray,
    config: DatasetConfig,
) -> None:
    unstable_fraction = 1.0 - float(samples["stable"].mean())
    if unstable_fraction > config.max_unstable_fraction_abort:
        raise RuntimeError(
            f"Dataset unstable fraction {unstable_fraction:.3f} exceeded limit "
            f"{config.max_unstable_fraction_abort:.3f}"
        )

    stable_mask = samples["stable"].to_numpy(dtype=bool)
    stable_trajectories = trajectories[stable_mask]
    if stable_trajectories.size and not np.all(np.isfinite(stable_trajectories)):
        raise RuntimeError("Stable trajectories contain NaN or infinite values.")
    if stable_trajectories.size and np.max(np.abs(stable_trajectories)) > config.max_abs_trajectory:
        raise RuntimeError("Stable trajectories exceeded the configured safe amplitude limit.")

    nonfinite_failures = int((samples["failure_reason"] == "non_finite_response").sum())
    if nonfinite_failures > 0:
        raise RuntimeError("Dataset generation produced non-finite responses.")


def _build_manifest(config: DatasetConfig) -> dict[str, Any]:
    payload = asdict(config)
    config_hash = hashlib.sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]
    return {
        "format_version": 2,
        "config_hash": config_hash,
        "config": payload,
        "total_chunks": math.ceil(config.n_plants / config.chunk_size_plants),
    }


def _validate_or_initialize_dataset_dir(
    dataset_dir: Path,
    chunks_dir: Path,
    manifest: dict[str, Any],
) -> None:
    manifest_path = dataset_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if existing["config_hash"] != manifest["config_hash"]:
            raise RuntimeError(
                f"Dataset directory {dataset_dir} already exists with a different configuration. "
                "Use a new dataset name."
            )
        expected_indices = set(range(int(manifest["total_chunks"])))
        actual_indices = {
            _chunk_index_from_path(path) for path in chunks_dir.glob("metadata_*.parquet")
        }
        if actual_indices and not actual_indices.issubset(expected_indices):
            raise RuntimeError(
                f"Dataset directory {dataset_dir} contains stale chunk files outside the "
                "expected range."
            )
    else:
        dump_json(manifest, manifest_path)


def _dataset_metadata(
    config: DatasetConfig,
    samples: pd.DataFrame,
    trajectories: np.ndarray,
) -> dict[str, Any]:
    split_counts = samples["split"].value_counts().sort_index().to_dict()
    stable_counts = (
        samples.groupby("split", dropna=False)["stable"].mean().mul(100.0).round(2).to_dict()
    )
    family_counts = samples["plant_family"].value_counts().sort_index().to_dict()
    family_stable = (
        samples.groupby("plant_family", dropna=False)["stable"].mean().mul(100.0).round(2).to_dict()
    )
    failure_reason_counts = (
        samples.loc[samples["failure_reason"] != "", "failure_reason"].value_counts().to_dict()
    )
    stable_mask = samples["stable"].to_numpy(dtype=bool)
    stable_trajectories = trajectories[stable_mask]
    trajectory_peak_abs = samples.loc[samples["stable"], "trajectory_peak_abs"].to_numpy(
        dtype=float
    )
    trajectory_stats = {
        "mean_peak_abs": (
            float(np.nanmean(trajectory_peak_abs)) if trajectory_peak_abs.size else float("nan")
        ),
        "p95_peak_abs": float(np.nanpercentile(trajectory_peak_abs, 95.0))
        if trajectory_peak_abs.size
        else float("nan"),
        "max_peak_abs": (
            float(np.nanmax(trajectory_peak_abs)) if trajectory_peak_abs.size else float("nan")
        ),
        "min_value": (
            float(np.nanmin(stable_trajectories)) if stable_trajectories.size else float("nan")
        ),
        "max_value": (
            float(np.nanmax(stable_trajectories)) if stable_trajectories.size else float("nan")
        ),
    }
    gain_mode_counts = samples["gain_sampling_mode"].value_counts().sort_index().to_dict()
    return {
        "name": config.name,
        "n_samples": int(samples.shape[0]),
        "n_plants": int(samples["plant_id"].nunique()),
        "n_time_steps": int(trajectories.shape[1]),
        "stable_fraction_pct": round(float(samples["stable"].mean() * 100.0), 2),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "split_stable_fraction_pct": {key: float(value) for key, value in stable_counts.items()},
        "family_counts": {key: int(value) for key, value in family_counts.items()},
        "family_stable_fraction_pct": {key: float(value) for key, value in family_stable.items()},
        "failure_reason_counts": {key: int(value) for key, value in failure_reason_counts.items()},
        "gain_sampling_mode_counts": {key: int(value) for key, value in gain_mode_counts.items()},
        "trajectory_stats": trajectory_stats,
        "ood_families": list(config.ood_families),
        "families": list(config.families),
        "controllers_per_plant": config.controllers_per_plant,
        "time_horizon": config.t_final,
        "num_workers": config.num_workers,
    }


def _write_dataset_diagnostic_plots(
    config: DatasetConfig,
    dataset_dir: Path,
    samples: pd.DataFrame,
    trajectories: np.ndarray,
) -> None:
    plots_root = dataset_dir.parent.parent / "plots"
    plots_dir = ensure_dir(plots_root / config.name)
    stable_mask = samples["stable"].to_numpy(dtype=bool)
    stable_trajectories = trajectories[stable_mask]

    plot_dataset_family_stability(
        samples,
        output_path=plots_dir / "family_stability.png",
        title=f"Stable Fraction By Family: {config.name}",
    )
    plot_gain_distributions(
        samples,
        output_path=plots_dir / "gain_distributions.png",
        title=f"PID Gain Distributions: {config.name}",
    )
    plot_trajectory_amplitudes(
        samples.loc[samples["stable"], "trajectory_peak_abs"].to_numpy(dtype=float),
        output_path=plots_dir / "trajectory_peak_abs.png",
        title=f"Stable Trajectory Peak Amplitudes: {config.name}",
    )

    diagnostics = {
        "plots_dir": str(plots_dir),
        "stable_trajectory_count": int(stable_trajectories.shape[0]),
        "nan_placeholder_trajectory_count": int((~stable_mask).sum()),
    }
    dump_json(diagnostics, dataset_dir / "diagnostics.json")


def _chunk_metadata_path(chunks_dir: Path, chunk_index: int) -> Path:
    return chunks_dir / f"metadata_{chunk_index:04d}.parquet"


def _chunk_trajectory_path(chunks_dir: Path, chunk_index: int) -> Path:
    return chunks_dir / f"trajectories_{chunk_index:04d}.npz"


def _chunk_index_from_path(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _directory_size_bytes(path: Path) -> int:
    return sum(file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file())


def _gain_bounds(
    plant: Plant,
    config: DatasetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_scales = np.asarray(heuristic_pid_scales(plant), dtype=np.float64)
    low_multipliers = np.asarray(
        [
            config.kp_multiplier_range[0],
            config.ki_multiplier_range[0],
            config.kd_multiplier_range[0],
        ],
        dtype=np.float64,
    )
    high_multipliers = np.asarray(
        [
            config.kp_multiplier_range[1],
            config.ki_multiplier_range[1],
            config.kd_multiplier_range[1],
        ],
        dtype=np.float64,
    )
    return base_scales * low_multipliers, base_scales * high_multipliers, base_scales
