from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from itertools import repeat
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from controlsimulator.config import DatasetConfig, save_config_snapshot
from controlsimulator.metrics import extract_response_metrics
from controlsimulator.plants import (
    Plant,
    heuristic_pid_scales,
    sample_pid_gains,
    sample_plant,
)
from controlsimulator.plotting import (
    plot_class_balance,
    plot_dataset_family_stability,
    plot_gain_distributions,
    plot_oscillation_frequency_distribution,
    plot_pole_distribution,
    plot_trajectory_amplitudes,
)
from controlsimulator.simulate import (
    closed_loop_characteristics,
    closed_loop_is_stable,
    simulate_closed_loop,
)
from controlsimulator.splits import assert_no_plant_leakage, assign_dataset_splits
from controlsimulator.utils import dump_json, ensure_dir, load_json, resolve_path

MANIFEST_FILENAME = "manifest.json"
PLANT_SPLITS_FILENAME = "plant_splits.parquet"
SAMPLES_FILENAME = "samples.parquet"
TRAJECTORIES_FILENAME = "trajectories.npz"
EXPORT_DIRNAME = "dataset"
_SPLIT_MAP_CACHE: dict[str, dict[int, str]] = {}


@dataclass(slots=True)
class DatasetBundle:
    samples: pd.DataFrame
    trajectories: np.ndarray
    time_grid: np.ndarray
    metadata: dict[str, Any]
    dataset_dir: Path


@dataclass(slots=True)
class DatasetChunk:
    chunk_index: int
    metadata_path: Path
    trajectory_path: Path


def time_grid_from_config(config: DatasetConfig) -> np.ndarray:
    return np.linspace(0.0, config.t_final, config.n_time_steps, dtype=np.float64)


def _trajectory_storage_dtype(config: DatasetConfig) -> np.dtype[np.floating[Any]]:
    if config.trajectory_storage_dtype == "float16":
        return np.dtype(np.float16)
    return np.dtype(np.float32)


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
            _validate_chunk_health(frame, config)
            frame.to_parquet(metadata_path, index=False, compression="zstd")
            np.savez_compressed(
                trajectory_path,
                trajectories=trajectories.astype(_trajectory_storage_dtype(config)),
            )

        _satisfy_response_quotas(
            config=config,
            dataset_dir=dataset_dir,
            chunks_dir=chunks_dir,
            time_grid=time_grid,
            executor=executor,
        )
    finally:
        if executor is not None:
            executor.shutdown()

    dataset_dir = consolidate_dataset(config)
    if config.export_dataset_layout:
        export_dataset_layout(dataset_dir)
    metadata = load_json(dataset_dir / "metadata.json")
    metadata["generation_seconds"] = perf_counter() - generation_start
    metadata["dataset_size_bytes"] = _directory_size_bytes(dataset_dir)
    dump_json(metadata, dataset_dir / "metadata.json")
    return dataset_dir


def consolidate_dataset(config: DatasetConfig) -> Path:
    dataset_dir = ensure_dir(resolve_path(config.dataset_dir()))
    chunks = dataset_chunks(dataset_dir)
    if not chunks:
        raise RuntimeError(f"No dataset chunks found in {dataset_dir}.")

    plant_splits = _build_plant_split_frame(chunks, config)
    split_map = dict(
        zip(
            plant_splits["plant_id"].astype(int).to_list(),
            plant_splits["split"].astype(str).to_list(),
            strict=False,
        )
    )
    plant_splits.to_parquet(dataset_dir / PLANT_SPLITS_FILENAME, index=False, compression="zstd")

    aggregate = _empty_aggregate_state()
    maybe_frames: list[pd.DataFrame] | None = []
    maybe_trajectories: list[np.ndarray] | None = []
    should_write_consolidated = (
        config.write_consolidated_outputs
        and (config.n_plants * config.controllers_per_plant) <= config.consolidation_sample_limit
    )
    if not should_write_consolidated:
        maybe_frames = None
        maybe_trajectories = None
        for stale_file in [dataset_dir / SAMPLES_FILENAME, dataset_dir / TRAJECTORIES_FILENAME]:
            if stale_file.exists():
                stale_file.unlink()

    for chunk in tqdm(chunks, desc=f"consolidate:{config.name}"):
        frame = _read_parquet_with_retries(chunk.metadata_path)
        frame["split"] = frame["plant_id"].map(split_map)
        if frame["split"].isna().any():
            raise RuntimeError(
                f"Missing split assignments for chunk {chunk.chunk_index} in {dataset_dir}."
            )

        assert_no_plant_leakage(frame[["plant_id", "split"]].drop_duplicates())

        trajectories = _load_chunk_trajectories(chunk.trajectory_path)
        if frame.shape[0] != trajectories.shape[0]:
            raise RuntimeError(
                f"Chunk {chunk.chunk_index} metadata row count does not match trajectories."
            )

        _update_aggregate_state(aggregate, frame, trajectories)

        if (
            should_write_consolidated
            and maybe_frames is not None
            and maybe_trajectories is not None
        ):
            maybe_frames.append(frame)
            maybe_trajectories.append(trajectories)

    _validate_dataset_health(aggregate, config)
    metadata = _aggregate_metadata(config, aggregate)
    dump_json(metadata, dataset_dir / "metadata.json")
    _write_dataset_diagnostic_plots(config, dataset_dir, aggregate)

    if should_write_consolidated and maybe_frames is not None and maybe_trajectories is not None:
        samples = pd.concat(maybe_frames, ignore_index=True)
        trajectories = np.concatenate(maybe_trajectories, axis=0).astype(np.float32)
        samples.to_parquet(dataset_dir / SAMPLES_FILENAME, index=False, compression="zstd")
        np.savez_compressed(dataset_dir / TRAJECTORIES_FILENAME, trajectories=trajectories)

    return dataset_dir


def load_dataset(dataset_dir: str | Path) -> DatasetBundle:
    resolved = resolve_path(dataset_dir)
    samples_path = resolved / SAMPLES_FILENAME
    trajectories_path = resolved / TRAJECTORIES_FILENAME
    if not samples_path.exists() or not trajectories_path.exists():
        metadata = load_json(resolved / "metadata.json")
        raise RuntimeError(
            f"Dataset {resolved} was generated in chunk-native mode and does not expose "
            f"consolidated arrays. Metadata reports {metadata['n_samples']} samples; "
            "use iter_dataset_chunks() for streamed access."
        )

    samples = _read_parquet_with_retries(samples_path)
    trajectories = np.load(trajectories_path)["trajectories"]
    time_grid = np.load(resolved / "time_grid.npy")
    metadata = load_json(resolved / "metadata.json")
    return DatasetBundle(
        samples=samples,
        trajectories=trajectories,
        time_grid=time_grid,
        metadata=metadata,
        dataset_dir=resolved,
    )


def dataset_chunks(dataset_dir: str | Path) -> list[DatasetChunk]:
    resolved = resolve_path(dataset_dir)
    manifest = load_json(resolved / MANIFEST_FILENAME)
    chunks_dir = resolved / "chunks"
    chunks: list[DatasetChunk] = []
    for chunk_index in range(int(manifest["total_chunks"])):
        metadata_path = _chunk_metadata_path(chunks_dir, chunk_index)
        trajectory_path = _chunk_trajectory_path(chunks_dir, chunk_index)
        if not metadata_path.exists() or not trajectory_path.exists():
            raise RuntimeError(
                f"Missing chunk files for chunk {chunk_index} in dataset {resolved}."
            )
        chunks.append(
            DatasetChunk(
                chunk_index=chunk_index,
                metadata_path=metadata_path,
                trajectory_path=trajectory_path,
            )
        )
    return chunks


def dataset_time_grid(dataset_dir: str | Path) -> np.ndarray:
    return np.load(resolve_path(dataset_dir) / "time_grid.npy")


def _read_parquet_with_retries(
    path: Path,
    *,
    columns: list[str] | None = None,
    retries: int = 6,
    backoff_seconds: float = 1.5,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return pd.read_parquet(path, columns=columns)
        except (TimeoutError, OSError, EOFError, ValueError, pa.ArrowException) as error:
            last_error = error
            if attempt + 1 >= retries:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    raise RuntimeError(f"Failed to read parquet after {retries} attempts: {path}") from last_error


def _load_chunk_trajectories(
    path: Path,
    *,
    retries: int = 6,
    backoff_seconds: float = 1.5,
) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with np.load(path) as payload:
                return payload["trajectories"].astype(np.float32)
        except (TimeoutError, OSError, EOFError, ValueError) as error:
            last_error = error
            if attempt + 1 >= retries:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    raise RuntimeError(
        f"Failed to load trajectory chunk after {retries} attempts: {path}"
    ) from last_error


def dataset_metadata(dataset_dir: str | Path) -> dict[str, Any]:
    return load_json(resolve_path(dataset_dir) / "metadata.json")


def _dataset_split_map(dataset_dir: Path) -> dict[int, str]:
    cache_key = str(dataset_dir)
    cached = _SPLIT_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    split_frame = _read_parquet_with_retries(
        dataset_dir / PLANT_SPLITS_FILENAME,
        columns=["plant_id", "split"],
    )
    split_map = dict(
        zip(
            split_frame["plant_id"].astype(int).to_list(),
            split_frame["split"].astype(str).to_list(),
            strict=False,
        )
    )
    _SPLIT_MAP_CACHE[cache_key] = split_map
    return split_map


def iter_dataset_chunks(
    dataset_dir: str | Path,
    *,
    include_trajectories: bool,
    splits: set[str] | None = None,
    stable_only: bool = False,
    columns: list[str] | None = None,
    progress_desc: str | None = None,
) -> Iterator[tuple[pd.DataFrame, np.ndarray | None]]:
    resolved = resolve_path(dataset_dir)
    required_columns = set(columns or [])
    need_split = splits is not None or "split" in required_columns
    if need_split:
        required_columns.add("plant_id")
        required_columns.discard("split")
    if stable_only:
        required_columns.add("stable")

    split_map = _dataset_split_map(resolved) if need_split else None

    chunks = dataset_chunks(resolved)
    chunk_iterator = (
        tqdm(chunks, desc=progress_desc, unit="chunk") if progress_desc is not None else chunks
    )

    for chunk in chunk_iterator:
        frame = _read_parquet_with_retries(
            chunk.metadata_path,
            columns=sorted(required_columns) if required_columns else None,
        )
        if need_split:
            frame["split"] = frame["plant_id"].map(split_map)
            if frame["split"].isna().any():
                raise RuntimeError(
                    "Missing split assignments for "
                    f"chunk {chunk.chunk_index} in dataset {resolved}."
                )
        mask = np.ones(frame.shape[0], dtype=bool)
        if splits is not None:
            mask &= frame["split"].isin(splits).to_numpy(dtype=bool)
        if stable_only:
            mask &= frame["stable"].to_numpy(dtype=bool)
        if not np.any(mask):
            continue

        trajectories = None
        if include_trajectories:
            trajectories = _load_chunk_trajectories(chunk.trajectory_path)
            trajectories = trajectories[mask]
        frame = frame.loc[mask].reset_index(drop=True)
        yield frame, trajectories


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
    plant = sample_plant(
        rng,
        plant_id=plant_id,
        families=config.families,
        family_sampling_weights=config.family_sampling_weights,
    )
    rows: list[dict[str, Any]] = []
    trajectories: list[np.ndarray] = []
    sampled_gains: list[dict[str, Any]] = []

    for controller_index, mode in enumerate(_sampling_modes(config)):
        if mode in {"random", "aggressive", "integral_heavy", "weak"}:
            kp, ki, kd = _sample_campaign_pid_gains(rng, mode)
            sampling_mode = mode
        elif mode == "wide_random":
            kp, ki, kd = sample_pid_gains(
                rng,
                plant,
                tuple(config.kp_multiplier_range),
                tuple(config.ki_multiplier_range),
                tuple(config.kd_multiplier_range),
            )
            sampling_mode = "wide_random"
        elif mode == "oscillatory_target":
            kp, ki, kd, sampling_mode = _sample_oscillatory_pid_gains(
                rng,
                plant,
                config,
                sampled_gains,
            )
        else:
            kp, ki, kd, sampling_mode = _sample_boundary_pid_gains(
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
            trajectory_clip_abs=config.trajectory_storage_clip_abs,
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
            "plant_min_damping_ratio": plant.min_damping_ratio,
            "plant_max_oscillation_hz": plant.max_oscillation_hz,
            "plant_pole_spread_log10": plant.pole_spread_log10,
            "plant_has_complex_poles": float(plant.has_complex_poles),
            "plant_max_real_part": plant.max_real_part,
            "plant_min_real_part": plant.min_real_part,
            "plant_root_stable": bool(plant.max_real_part < 0.0),
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "tau_d": config.derivative_filter_tau,
            "root_stable": bool(result.stability_margin < -1e-6),
            "stable": bool(is_usable),
            "stability_margin": result.stability_margin,
            "closed_loop_oscillation_hz": result.dominant_oscillation_hz,
            "closed_loop_min_damping_ratio": result.min_damping_ratio,
            "failure_reason": failure_reason,
            "gain_sampling_mode": sampling_mode,
            "trajectory_peak_abs": trajectory_peak_abs,
            "trajectory_final_value": final_value,
            "response_is_unstable": bool(not is_usable),
            "response_is_near_instability": bool(
                abs(float(result.stability_margin)) <= config.near_instability_margin
            ),
        }

        padded_num = plant.padded_numerator()
        padded_den = plant.padded_denominator()
        for index, value in enumerate(padded_num):
            row[f"num_{index}"] = float(value)
        for index, value in enumerate(padded_den):
            row[f"den_{index}"] = float(value)
        row["b0"] = float(padded_num[-1])
        row["a2"] = float(padded_den[-3])
        row["a1"] = float(padded_den[-2])
        row["a0"] = float(padded_den[-1])
        sorted_poles = sorted(plant.poles, key=lambda pole: (np.real(pole), np.imag(pole)))
        for pole_index, pole in enumerate(sorted_poles):
            row[f"pole_{pole_index}_real"] = float(np.real(pole))
            row[f"pole_{pole_index}_imag"] = float(np.imag(pole))
        for pole_index in range(len(sorted_poles), 3):
            row[f"pole_{pole_index}_real"] = np.nan
            row[f"pole_{pole_index}_imag"] = np.nan

        if result.trajectory is not None:
            response_metrics = extract_response_metrics(
                time_grid,
                result.trajectory,
                peak_control_effort=result.peak_control_effort,
            )
            row.update(response_metrics.to_dict())
            row["response_is_oscillatory"] = bool(
                (
                    np.isfinite(response_metrics.oscillation_frequency_estimate_hz)
                    and response_metrics.oscillation_frequency_estimate_hz
                    >= config.oscillatory_frequency_threshold_hz
                )
                or result.dominant_oscillation_hz >= config.oscillatory_frequency_threshold_hz
            )
            trajectories.append(result.trajectory.astype(np.float32))
        else:
            row.update(
                {
                    "overshoot_pct": np.nan,
                    "rise_time": np.nan,
                    "settling_time": np.nan,
                    "steady_state_error": np.nan,
                    "oscillation_frequency_estimate_hz": np.nan,
                    "peak_control_effort": np.nan,
                }
            )
            row["response_is_oscillatory"] = bool(
                result.dominant_oscillation_hz >= config.oscillatory_frequency_threshold_hz
            )
            trajectories.append(np.full_like(time_grid, np.nan, dtype=np.float32))

        rows.append(row)
        sampled_gains.append(
            {
                "kp": kp,
                "ki": ki,
                "kd": kd,
                "stable": bool(is_usable),
                "stability_margin": float(result.stability_margin),
                "closed_loop_oscillation_hz": float(result.dominant_oscillation_hz),
                "closed_loop_min_damping_ratio": float(result.min_damping_ratio),
            }
        )

    return rows, trajectories


def _sampling_modes(config: DatasetConfig) -> list[str]:
    if config.controller_mode_weights:
        return _weighted_modes(config.controller_mode_weights, config.controllers_per_plant)
    fractions = {
        "wide_random": config.wide_sampling_fraction,
        "boundary_search": float(config.boundary_sampling_fraction),
        "oscillatory_target": config.oscillatory_sampling_fraction,
    }
    total_controllers = config.controllers_per_plant
    raw_counts = {key: value * total_controllers for key, value in fractions.items()}
    counts = {key: int(math.floor(value)) for key, value in raw_counts.items()}
    remainder = total_controllers - sum(counts.values())
    residuals = sorted(
        ((raw_counts[key] - counts[key], key) for key in fractions),
        reverse=True,
    )
    for _, key in residuals[:remainder]:
        counts[key] += 1

    modes: list[str] = []
    for key in ["wide_random", "boundary_search", "oscillatory_target"]:
        modes.extend([key] * counts[key])
    return modes


def _weighted_modes(weights: dict[str, float], total_controllers: int) -> list[str]:
    raw_counts = {key: max(float(value), 0.0) * total_controllers for key, value in weights.items()}
    counts = {key: int(math.floor(value)) for key, value in raw_counts.items()}
    remainder = total_controllers - sum(counts.values())
    residuals = sorted(
        ((raw_counts[key] - counts[key], key) for key in weights),
        reverse=True,
    )
    for _, key in residuals[:remainder]:
        counts[key] += 1

    modes: list[str] = []
    for key in weights:
        modes.extend([key] * counts[key])
    return modes


def _sample_campaign_pid_gains(
    rng: np.random.Generator,
    mode: str,
) -> tuple[float, float, float]:
    ranges = {
        "random": ((1e-2, 1e2), (1e-3, 1e2), (1e-4, 1e1)),
        "aggressive": ((1.0, 1e2), (1e-3, 3.0), (5e-2, 1e1)),
        "integral_heavy": ((1e-2, 1e1), (1.0, 1e2), (1e-4, 1.0)),
        "weak": ((1e-2, 3e-1), (1e-3, 1.0), (1e-4, 2e-1)),
    }
    try:
        kp_range, ki_range, kd_range = ranges[mode]
    except KeyError as error:
        raise ValueError(f"Unknown campaign controller sampling mode: {mode}") from error
    return (
        float(np.exp(rng.uniform(np.log(kp_range[0]), np.log(kp_range[1])))),
        float(np.exp(rng.uniform(np.log(ki_range[0]), np.log(ki_range[1])))),
        float(np.exp(rng.uniform(np.log(kd_range[0]), np.log(kd_range[1])))),
    )


def _sample_boundary_pid_gains(
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
        return kp, ki, kd, "boundary_fallback_random"

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


def _sample_oscillatory_pid_gains(
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
        return _sample_boundary_pid_gains(rng, plant, config, sampled_gains)

    stable_log = np.log(np.clip(stable_anchor, low_bounds, high_bounds))
    unstable_log = np.log(np.clip(unstable_anchor, low_bounds, high_bounds))
    best_stable: tuple[float, np.ndarray] | None = None
    best_unstable: tuple[float, np.ndarray] | None = None

    for _ in range(config.oscillatory_candidate_count):
        alpha = float(
            np.clip(
                rng.normal(
                    config.oscillatory_boundary_bias,
                    config.oscillatory_boundary_std,
                ),
                0.45,
                0.98,
            )
        )
        sample_log = ((1.0 - alpha) * stable_log) + (alpha * unstable_log)
        sample_log += rng.normal(0.0, config.oscillatory_jitter_std, size=3)
        candidate = np.clip(np.exp(sample_log), low_bounds, high_bounds)
        characteristics = closed_loop_characteristics(
            plant,
            float(candidate[0]),
            float(candidate[1]),
            float(candidate[2]),
            config.derivative_filter_tau,
        )
        boundary_proximity = 1.0 / max(abs(characteristics.stability_margin), 1e-4)
        score = (
            1.6 * np.log1p(characteristics.dominant_oscillation_hz)
            + 1.1 * (1.0 - min(characteristics.min_damping_ratio, 1.0))
            + 0.25 * np.log1p(plant.max_oscillation_hz)
            + 0.35 * boundary_proximity
        )
        if characteristics.stability_margin < -1e-6:
            if best_stable is None or score > best_stable[0]:
                best_stable = (score, candidate)
        else:
            if best_unstable is None or score > best_unstable[0]:
                best_unstable = (score, candidate)

    if best_stable is not None:
        candidate = best_stable[1]
        return float(candidate[0]), float(candidate[1]), float(candidate[2]), "oscillatory_target"
    if best_unstable is not None:
        candidate = best_unstable[1]
        return (
            float(candidate[0]),
            float(candidate[1]),
            float(candidate[2]),
            "oscillatory_unstable_target",
        )
    return _sample_boundary_pid_gains(rng, plant, config, sampled_gains)


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

    for scale in [0.8, 0.6, 0.45, 0.3, 0.2, 0.1, 0.05]:
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
    for _ in range(16):
        source = candidate_sources[int(rng.integers(0, len(candidate_sources)))]
        source_gains = np.asarray([source["kp"], source["ki"], source["kd"]], dtype=np.float64)
        amplification = np.exp(np.abs(rng.normal(loc=0.9, scale=0.55, size=3)))
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
    if result.trajectory is not None:
        trajectory_peak_abs = float(np.max(np.abs(result.trajectory)))
        final_value = float(result.trajectory[-1])
    else:
        trajectory_peak_abs = float("nan")
        final_value = float("nan")
    if not result.stable or result.trajectory is None:
        return False, result.reason or "unstable_closed_loop", trajectory_peak_abs, final_value
    if result.trajectory_clipped:
        return False, result.reason or "trajectory_clipped", trajectory_peak_abs, final_value

    if not np.isfinite(trajectory_peak_abs) or not np.isfinite(final_value):
        return False, "non_finite_response", trajectory_peak_abs, final_value
    if trajectory_peak_abs > config.max_abs_trajectory:
        return False, "trajectory_limit_exceeded", trajectory_peak_abs, final_value
    return True, "", trajectory_peak_abs, final_value


def _validate_chunk_health(frame: pd.DataFrame, config: DatasetConfig) -> None:
    _assert_chunk_has_no_forbidden_failures(frame)
    if config.controller_mode_weights is not None or _response_quotas(config):
        return
    unstable_fraction = 1.0 - float(frame["stable"].mean())
    if unstable_fraction > config.max_unstable_fraction_abort:
        raise RuntimeError(
            f"Chunk unstable fraction exceeded safety threshold: {unstable_fraction:.3f}"
        )


def _assert_chunk_has_no_forbidden_failures(frame: pd.DataFrame) -> None:
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


def _build_manifest(config: DatasetConfig) -> dict[str, Any]:
    payload = asdict(config)
    config_hash = hashlib.sha256(repr(sorted(payload.items())).encode("utf-8")).hexdigest()[:16]
    return {
        "format_version": 4,
        "config_hash": config_hash,
        "config": payload,
        "base_n_plants": int(config.n_plants),
        "next_plant_id": int(config.n_plants),
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


def _build_plant_split_frame(chunks: list[DatasetChunk], config: DatasetConfig) -> pd.DataFrame:
    plant_frames = []
    for chunk in chunks:
        plant_frames.append(
            _read_parquet_with_retries(
                chunk.metadata_path,
                columns=["plant_id", "plant_family"],
            ).drop_duplicates()
        )
    plants = pd.concat(plant_frames, ignore_index=True).drop_duplicates().sort_values("plant_id")
    plants["split"] = assign_dataset_splits(
        plants,
        val_fraction=config.val_fraction,
        test_fraction=config.test_fraction,
        ood_families=config.ood_families,
        seed=config.seed,
    )
    assert_no_plant_leakage(plants[["plant_id", "split"]])
    return plants.reset_index(drop=True)


def _empty_aggregate_state() -> dict[str, Any]:
    return {
        "n_samples": 0,
        "n_plants": set(),
        "n_time_steps": None,
        "split_counts": Counter(),
        "split_stable_sum": Counter(),
        "family_counts": Counter(),
        "family_stable_sum": Counter(),
        "response_category_counts": Counter(),
        "failure_reason_counts": Counter(),
        "gain_mode_counts": Counter(),
        "stable_peak_values": [],
        "stable_oscillation_hz": [],
        "stable_closed_loop_damping": [],
        "gain_sample_frames": [],
        "plant_sample_frames": [],
        "stable_value_min": float("inf"),
        "stable_value_max": float("-inf"),
        "stable_nan_count": 0,
    }


def _update_aggregate_state(
    aggregate: dict[str, Any],
    frame: pd.DataFrame,
    trajectories: np.ndarray,
) -> None:
    aggregate["n_samples"] += int(frame.shape[0])
    aggregate["n_plants"].update(frame["plant_id"].astype(int).tolist())
    aggregate["n_time_steps"] = int(trajectories.shape[1])

    split_counts = frame["split"].value_counts().to_dict()
    for split, count in split_counts.items():
        aggregate["split_counts"][split] += int(count)
        stable_fraction = frame.loc[frame["split"] == split, "stable"].sum()
        aggregate["split_stable_sum"][split] += int(stable_fraction)

    family_counts = frame["plant_family"].value_counts().to_dict()
    for family, count in family_counts.items():
        aggregate["family_counts"][family] += int(count)
        stable_count = frame.loc[frame["plant_family"] == family, "stable"].sum()
        aggregate["family_stable_sum"][family] += int(stable_count)

    aggregate["response_category_counts"].update(_response_category_counts(frame))

    failures = frame.loc[frame["failure_reason"] != "", "failure_reason"].value_counts().to_dict()
    for reason, count in failures.items():
        aggregate["failure_reason_counts"][reason] += int(count)

    modes = frame["gain_sampling_mode"].value_counts().to_dict()
    for mode, count in modes.items():
        aggregate["gain_mode_counts"][mode] += int(count)

    stable_mask = frame["stable"].to_numpy(dtype=bool)
    if np.any(stable_mask):
        stable_trajectories = trajectories[stable_mask]
        aggregate["stable_nan_count"] += int(np.isnan(stable_trajectories).sum())
        aggregate["stable_value_min"] = min(
            aggregate["stable_value_min"],
            float(np.nanmin(stable_trajectories)),
        )
        aggregate["stable_value_max"] = max(
            aggregate["stable_value_max"],
            float(np.nanmax(stable_trajectories)),
        )
        aggregate["stable_peak_values"].append(
            frame.loc[stable_mask, "trajectory_peak_abs"].to_numpy(dtype=np.float32)
        )
        aggregate["stable_oscillation_hz"].append(
            frame.loc[stable_mask, "closed_loop_oscillation_hz"].to_numpy(dtype=np.float32)
        )
        aggregate["stable_closed_loop_damping"].append(
            frame.loc[stable_mask, "closed_loop_min_damping_ratio"].to_numpy(dtype=np.float32)
        )

    gain_frame = frame[
        [
            "kp",
            "ki",
            "kd",
            "stable",
            "gain_sampling_mode",
            "plant_family",
            "plant_order",
            "plant_min_damping_ratio",
            "closed_loop_oscillation_hz",
        ]
    ]
    if gain_frame.shape[0] > 2_000:
        gain_frame = gain_frame.sample(2_000, random_state=42 + int(frame["plant_id"].iloc[0]))
    aggregate["gain_sample_frames"].append(gain_frame.reset_index(drop=True))
    plant_frame = frame[
        [
            "plant_id",
            "plant_family",
            "plant_root_stable",
            "plant_has_complex_poles",
            "plant_max_real_part",
            "plant_min_real_part",
            "pole_0_real",
            "pole_0_imag",
            "pole_1_real",
            "pole_1_imag",
            "pole_2_real",
            "pole_2_imag",
        ]
    ].drop_duplicates("plant_id")
    if plant_frame.shape[0] > 1_000:
        plant_frame = plant_frame.sample(1_000, random_state=84 + int(frame["plant_id"].iloc[0]))
    aggregate["plant_sample_frames"].append(plant_frame.reset_index(drop=True))


def _validate_dataset_health(aggregate: dict[str, Any], config: DatasetConfig) -> None:
    unstable_fraction = 1.0 - (
        sum(aggregate["split_stable_sum"].values()) / max(aggregate["n_samples"], 1)
    )
    if unstable_fraction > config.max_unstable_fraction_abort:
        raise RuntimeError(
            f"Dataset unstable fraction {unstable_fraction:.3f} exceeded limit "
            f"{config.max_unstable_fraction_abort:.3f}"
        )
    if aggregate["stable_nan_count"] > 0:
        raise RuntimeError("Stable trajectories contain NaN values.")
    if aggregate["stable_value_max"] > config.max_abs_trajectory:
        raise RuntimeError("Stable trajectories exceeded the configured safe amplitude limit.")
    if aggregate["failure_reason_counts"].get("non_finite_response", 0) > 0:
        raise RuntimeError("Dataset generation produced non-finite responses.")


def _aggregate_metadata(config: DatasetConfig, aggregate: dict[str, Any]) -> dict[str, Any]:
    stable_count = int(sum(aggregate["split_stable_sum"].values()))
    trajectory_peak_abs = _concatenate_float_arrays(aggregate["stable_peak_values"])
    stable_oscillation_hz = _concatenate_float_arrays(aggregate["stable_oscillation_hz"])
    stable_damping = _concatenate_float_arrays(aggregate["stable_closed_loop_damping"])
    gain_sample = (
        pd.concat(aggregate["gain_sample_frames"], ignore_index=True)
        if aggregate["gain_sample_frames"]
        else pd.DataFrame()
    )
    plant_sample = (
        pd.concat(aggregate["plant_sample_frames"], ignore_index=True)
        if aggregate["plant_sample_frames"]
        else pd.DataFrame()
    )

    split_stable_fraction = {}
    for split, count in aggregate["split_counts"].items():
        split_stable_fraction[split] = round(
            (aggregate["split_stable_sum"][split] / max(count, 1)) * 100.0,
            2,
        )

    family_stable_fraction = {}
    for family, count in aggregate["family_counts"].items():
        family_stable_fraction[family] = round(
            (aggregate["family_stable_sum"][family] / max(count, 1)) * 100.0,
            2,
        )

    return {
        "name": config.name,
        "n_samples": int(aggregate["n_samples"]),
        "n_plants": int(len(aggregate["n_plants"])),
        "n_time_steps": int(aggregate["n_time_steps"] or config.n_time_steps),
        "stable_fraction_pct": round((stable_count / max(aggregate["n_samples"], 1)) * 100.0, 2),
        "unstable_count": int(aggregate["response_category_counts"]["unstable"]),
        "oscillatory_count": int(aggregate["response_category_counts"]["oscillatory"]),
        "near_instability_count": int(aggregate["response_category_counts"]["near_instability"]),
        "split_counts": {key: int(value) for key, value in aggregate["split_counts"].items()},
        "split_stable_fraction_pct": split_stable_fraction,
        "family_counts": {key: int(value) for key, value in aggregate["family_counts"].items()},
        "family_stable_fraction_pct": family_stable_fraction,
        "failure_reason_counts": {
            key: int(value) for key, value in aggregate["failure_reason_counts"].items()
        },
        "response_category_counts": {
            key: int(value) for key, value in aggregate["response_category_counts"].items()
        },
        "gain_sampling_mode_counts": {
            key: int(value) for key, value in aggregate["gain_mode_counts"].items()
        },
        "trajectory_stats": {
            "mean_peak_abs": float(np.nanmean(trajectory_peak_abs))
            if trajectory_peak_abs.size
            else float("nan"),
            "p95_peak_abs": float(np.nanpercentile(trajectory_peak_abs, 95.0))
            if trajectory_peak_abs.size
            else float("nan"),
            "max_peak_abs": float(np.nanmax(trajectory_peak_abs))
            if trajectory_peak_abs.size
            else float("nan"),
            "min_value": float(aggregate["stable_value_min"])
            if np.isfinite(aggregate["stable_value_min"])
            else float("nan"),
            "max_value": float(aggregate["stable_value_max"])
            if np.isfinite(aggregate["stable_value_max"])
            else float("nan"),
            "oscillation_frequency_hz_p95": float(np.nanpercentile(stable_oscillation_hz, 95.0))
            if stable_oscillation_hz.size
            else float("nan"),
            "oscillation_frequency_hz_max": float(np.nanmax(stable_oscillation_hz))
            if stable_oscillation_hz.size
            else float("nan"),
            "closed_loop_damping_p05": float(np.nanpercentile(stable_damping, 5.0))
            if stable_damping.size
            else float("nan"),
        },
        "gain_stats": _sampled_gain_statistics(gain_sample),
        "pole_stats": _sampled_pole_statistics(plant_sample),
        "ood_families": list(config.ood_families),
        "families": list(config.families),
        "controllers_per_plant": config.controllers_per_plant,
        "time_horizon": config.t_final,
        "num_workers": config.num_workers,
        "write_consolidated_outputs": bool(
            config.write_consolidated_outputs
            and (
                config.n_plants * config.controllers_per_plant
                <= config.consolidation_sample_limit
            )
        ),
    }


def _write_dataset_diagnostic_plots(
    config: DatasetConfig,
    dataset_dir: Path,
    aggregate: dict[str, Any],
) -> None:
    plots_root = dataset_dir.parent.parent / "plots"
    plots_dir = ensure_dir(plots_root / config.name)
    sample_frame = (
        pd.concat(aggregate["gain_sample_frames"], ignore_index=True)
        if aggregate["gain_sample_frames"]
        else pd.DataFrame()
    )
    plant_frame = (
        pd.concat(aggregate["plant_sample_frames"], ignore_index=True)
        if aggregate["plant_sample_frames"]
        else pd.DataFrame()
    )
    family_summary = pd.DataFrame(
        {
            "plant_family": list(aggregate["family_counts"].keys()),
            "stable_fraction_pct": [
                (aggregate["family_stable_sum"][family] / max(count, 1)) * 100.0
                for family, count in aggregate["family_counts"].items()
            ],
        }
    )
    peak_values = _concatenate_float_arrays(aggregate["stable_peak_values"])
    oscillation_hz = _concatenate_float_arrays(aggregate["stable_oscillation_hz"])

    if not family_summary.empty:
        plot_dataset_family_stability(
            family_summary,
            output_path=plots_dir / "family_stability.png",
            title=f"Stable Fraction By Family: {config.name}",
        )
    if not sample_frame.empty:
        plot_gain_distributions(
            sample_frame,
            output_path=plots_dir / "gain_distributions.png",
            title=f"PID Gain Distributions: {config.name}",
        )
        plot_class_balance(
            stable_count=int(sum(aggregate["split_stable_sum"].values())),
            unstable_count=int(aggregate["response_category_counts"]["unstable"]),
            output_path=plots_dir / "class_balance.png",
            title=f"Stable vs Unstable Responses: {config.name}",
        )
    if not plant_frame.empty:
        plot_pole_distribution(
            plant_frame,
            output_path=plots_dir / "pole_distribution.png",
            title=f"Plant Pole Distribution: {config.name}",
        )
    plot_trajectory_amplitudes(
        peak_values,
        output_path=plots_dir / "trajectory_peak_abs.png",
        title=f"Stable Trajectory Peak Amplitudes: {config.name}",
    )
    plot_oscillation_frequency_distribution(
        oscillation_hz,
        output_path=plots_dir / "oscillation_frequency_hz.png",
        title=f"Stable Closed-Loop Oscillation Frequency: {config.name}",
    )

    diagnostics = {
        "plots_dir": str(plots_dir),
        "stable_trajectory_count": int(sum(aggregate["split_stable_sum"].values())),
        "nan_placeholder_trajectory_count": int(
            aggregate["n_samples"] - sum(aggregate["split_stable_sum"].values())
        ),
        "response_category_counts": {
            key: int(value) for key, value in aggregate["response_category_counts"].items()
        },
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


def _response_quotas(config: DatasetConfig) -> dict[str, int]:
    quotas = {
        "unstable": config.unstable_response_target,
        "oscillatory": config.oscillatory_response_target,
        "near_instability": config.near_instability_response_target,
    }
    return {key: int(value) for key, value in quotas.items() if value is not None and value > 0}


def _response_category_counts(frame: pd.DataFrame) -> Counter[str]:
    return Counter(
        {
            "unstable": int((~frame["stable"].to_numpy(dtype=bool)).sum()),
            "oscillatory": int(frame["response_is_oscillatory"].to_numpy(dtype=bool).sum()),
            "near_instability": int(
                frame["response_is_near_instability"].to_numpy(dtype=bool).sum()
            ),
        }
    )


def _count_existing_response_categories(dataset_dir: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    for chunk in dataset_chunks(dataset_dir):
        frame = _read_parquet_with_retries(
            chunk.metadata_path,
            columns=["stable", "response_is_oscillatory", "response_is_near_instability"],
        )
        counts.update(_response_category_counts(frame))
    return counts


def _top_up_config_for_deficit(config: DatasetConfig, category: str) -> DatasetConfig:
    profiles = {
        "unstable": {
            "families": [
                "campaign_third_order_unstable",
                "campaign_third_order_near_instability",
            ],
            "family_sampling_weights": {
                "campaign_third_order_unstable": 0.7,
                "campaign_third_order_near_instability": 0.3,
            },
            "controller_mode_weights": {"aggressive": 0.7, "random": 0.2, "integral_heavy": 0.1},
        },
        "oscillatory": {
            "families": [
                "campaign_third_order_oscillatory",
                "campaign_third_order_ood_lightly_damped",
            ],
            "family_sampling_weights": {
                "campaign_third_order_oscillatory": 0.65,
                "campaign_third_order_ood_lightly_damped": 0.35,
            },
            "controller_mode_weights": {"aggressive": 0.55, "integral_heavy": 0.3, "random": 0.15},
        },
        "near_instability": {
            "families": [
                "campaign_third_order_near_instability",
                "campaign_third_order_oscillatory",
            ],
            "family_sampling_weights": {
                "campaign_third_order_near_instability": 0.75,
                "campaign_third_order_oscillatory": 0.25,
            },
            "controller_mode_weights": {"integral_heavy": 0.45, "aggressive": 0.35, "weak": 0.2},
        },
    }
    try:
        overrides = profiles[category]
    except KeyError as error:
        raise ValueError(f"Unsupported response-quota category: {category}") from error
    return replace(config, **overrides)


def _satisfy_response_quotas(
    config: DatasetConfig,
    dataset_dir: Path,
    chunks_dir: Path,
    time_grid: np.ndarray,
    executor: ProcessPoolExecutor | None,
) -> None:
    quotas = _response_quotas(config)
    if not quotas:
        return

    manifest_path = dataset_dir / MANIFEST_FILENAME
    manifest = load_json(manifest_path)
    counts = _count_existing_response_categories(dataset_dir)
    next_chunk_index = int(manifest["total_chunks"])
    next_plant_id = int(manifest.get("next_plant_id", config.n_plants))
    rounds = 0

    while any(counts.get(key, 0) < target for key, target in quotas.items()):
        if rounds >= config.quota_max_rounds:
            unmet = {
                key: int(target - counts.get(key, 0))
                for key, target in quotas.items()
                if counts.get(key, 0) < target
            }
            raise RuntimeError(f"Unable to satisfy response quotas within max rounds: {unmet}")

        deficits = {
            key: int(target - counts.get(key, 0))
            for key, target in quotas.items()
            if counts.get(key, 0) < target
        }
        category = max(deficits, key=deficits.get)
        plants_needed = max(
            config.chunk_size_plants,
            min(
                config.quota_resample_batch_plants,
                math.ceil((deficits[category] / max(config.controllers_per_plant, 1)) * 1.5),
            ),
        )
        target_config = _top_up_config_for_deficit(config, category)
        frame, trajectories = _generate_chunk(
            config=target_config,
            time_grid=time_grid,
            start_plant=next_plant_id,
            end_plant=next_plant_id + plants_needed,
            executor=executor,
        )
        _assert_chunk_has_no_forbidden_failures(frame)
        frame.to_parquet(
            _chunk_metadata_path(chunks_dir, next_chunk_index),
            index=False,
            compression="zstd",
        )
        np.savez_compressed(
            _chunk_trajectory_path(chunks_dir, next_chunk_index),
            trajectories=trajectories.astype(_trajectory_storage_dtype(config)),
        )
        counts.update(_response_category_counts(frame))
        next_chunk_index += 1
        next_plant_id += plants_needed
        rounds += 1
        manifest["total_chunks"] = next_chunk_index
        manifest["next_plant_id"] = next_plant_id
        manifest["response_quota_counts"] = {key: int(value) for key, value in counts.items()}
        manifest["quota_rounds"] = rounds
        dump_json(manifest, manifest_path)


def export_dataset_layout(dataset_dir: str | Path) -> Path:
    resolved = resolve_path(dataset_dir)
    export_dir = ensure_dir(resolved / EXPORT_DIRNAME)
    metadata = dataset_metadata(resolved)
    n_samples = int(metadata["n_samples"])
    n_time_steps = int(metadata["n_time_steps"])

    for filename in [
        "plants.parquet",
        "controllers.parquet",
        "metrics.parquet",
        "labels.parquet",
        "trajectories.npy",
    ]:
        target = export_dir / filename
        if target.exists():
            target.unlink()

    np.save(export_dir / "time_grid.npy", dataset_time_grid(resolved))
    trajectories_path = export_dir / "trajectories.npy"
    trajectory_array = np.lib.format.open_memmap(
        trajectories_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_samples, n_time_steps),
    )

    plant_columns = [
        "plant_id",
        "plant_family",
        "plant_order",
        "b0",
        "a2",
        "a1",
        "a0",
        "dc_gain",
        "dominant_pole_mag",
        "mean_pole_mag",
        "plant_min_damping_ratio",
        "plant_max_oscillation_hz",
        "plant_pole_spread_log10",
        "plant_has_complex_poles",
        "plant_max_real_part",
        "plant_min_real_part",
        "plant_root_stable",
        "pole_0_real",
        "pole_0_imag",
        "pole_1_real",
        "pole_1_imag",
        "pole_2_real",
        "pole_2_imag",
        "num_0",
        "num_1",
        "num_2",
        "den_0",
        "den_1",
        "den_2",
        "den_3",
        "den_4",
    ]
    controller_columns = [
        "sample_id",
        "plant_id",
        "controller_index",
        "kp",
        "ki",
        "kd",
        "tau_d",
        "gain_sampling_mode",
    ]
    metric_columns = [
        "sample_id",
        "trajectory_peak_abs",
        "trajectory_final_value",
        "stability_margin",
        "closed_loop_oscillation_hz",
        "closed_loop_min_damping_ratio",
        "overshoot_pct",
        "rise_time",
        "settling_time",
        "steady_state_error",
        "oscillation_frequency_estimate_hz",
        "peak_control_effort",
    ]
    label_columns = [
        "sample_id",
        "plant_id",
        "split",
        "stable",
        "root_stable",
        "plant_root_stable",
        "response_is_unstable",
        "response_is_oscillatory",
        "response_is_near_instability",
        "failure_reason",
    ]
    writers: dict[str, pq.ParquetWriter] = {}
    seen_plant_ids: set[int] = set()
    row_offset = 0

    try:
        for chunk in tqdm(
            dataset_chunks(resolved),
            desc=f"export:{resolved.name}",
            unit="chunk",
        ):
            frame = _read_parquet_with_retries(chunk.metadata_path)
            if "split" not in frame.columns:
                frame["split"] = frame["plant_id"].map(_dataset_split_map(resolved))
            trajectories = _load_chunk_trajectories(chunk.trajectory_path)
            next_offset = row_offset + frame.shape[0]
            trajectory_array[row_offset:next_offset] = trajectories
            row_offset = next_offset

            plant_frame = frame[plant_columns].drop_duplicates("plant_id")
            if seen_plant_ids:
                plant_frame = plant_frame.loc[
                    ~plant_frame["plant_id"].isin(seen_plant_ids)
                ].reset_index(drop=True)
            seen_plant_ids.update(plant_frame["plant_id"].astype(int).tolist())

            _write_export_parquet_chunk(
                export_dir / "plants.parquet",
                plant_frame,
                writers,
                "plants",
            )
            _write_export_parquet_chunk(
                export_dir / "controllers.parquet",
                frame[controller_columns],
                writers,
                "controllers",
            )
            _write_export_parquet_chunk(
                export_dir / "metrics.parquet",
                frame[metric_columns],
                writers,
                "metrics",
            )
            _write_export_parquet_chunk(
                export_dir / "labels.parquet",
                frame[label_columns],
                writers,
                "labels",
            )
    finally:
        trajectory_array.flush()
        for writer in writers.values():
            writer.close()

    dump_json(
        {
            "n_samples": n_samples,
            "n_time_steps": n_time_steps,
            "n_plants": int(metadata["n_plants"]),
            "export_dir": str(export_dir),
        },
        export_dir / "manifest.json",
    )
    return export_dir


def _write_export_parquet_chunk(
    path: Path,
    frame: pd.DataFrame,
    writers: dict[str, pq.ParquetWriter],
    key: str,
) -> None:
    if frame.empty:
        return
    table = pa.Table.from_pandas(frame, preserve_index=False)
    writer = writers.get(key)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
        writers[key] = writer
    writer.write_table(table)


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


def _concatenate_float_arrays(values: list[np.ndarray]) -> np.ndarray:
    if not values:
        return np.array([], dtype=np.float32)
    return np.concatenate(values).astype(np.float32)


def _sampled_gain_statistics(sample: pd.DataFrame) -> dict[str, float]:
    if sample.empty:
        return {}
    summary: dict[str, float] = {}
    for column in ["kp", "ki", "kd"]:
        values = sample[column].to_numpy(dtype=float)
        summary[f"{column}_p05"] = float(np.nanpercentile(values, 5.0))
        summary[f"{column}_median"] = float(np.nanpercentile(values, 50.0))
        summary[f"{column}_p95"] = float(np.nanpercentile(values, 95.0))
    return summary


def _sampled_pole_statistics(sample: pd.DataFrame) -> dict[str, float]:
    if sample.empty:
        return {}
    real_values: list[np.ndarray] = []
    imag_values: list[np.ndarray] = []
    for pole_index in range(3):
        real_values.append(sample[f"pole_{pole_index}_real"].to_numpy(dtype=float))
        imag_values.append(sample[f"pole_{pole_index}_imag"].to_numpy(dtype=float))
    real_parts = np.concatenate(real_values)
    imag_parts = np.concatenate(imag_values)
    return {
        "real_part_min": float(np.nanmin(real_parts)),
        "real_part_p50": float(np.nanpercentile(real_parts, 50.0)),
        "real_part_max": float(np.nanmax(real_parts)),
        "imag_part_abs_p95": float(np.nanpercentile(np.abs(imag_parts), 95.0)),
    }
