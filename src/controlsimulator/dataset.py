from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from controlsimulator.config import DatasetConfig, save_config_snapshot
from controlsimulator.metrics import extract_response_metrics
from controlsimulator.plants import sample_pid_gains, sample_plant
from controlsimulator.simulate import simulate_closed_loop
from controlsimulator.splits import assert_no_plant_leakage, assign_dataset_splits
from controlsimulator.utils import dump_json, ensure_dir, load_json, resolve_path


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
    save_config_snapshot(config, dataset_dir / "config_snapshot.yaml")

    time_grid = time_grid_from_config(config)
    np.save(dataset_dir / "time_grid.npy", time_grid)

    total_chunks = math.ceil(config.n_plants / config.chunk_size_plants)
    for chunk_index in tqdm(range(total_chunks), desc=f"dataset:{config.name}"):
        metadata_path = chunks_dir / f"metadata_{chunk_index:04d}.csv"
        trajectory_path = chunks_dir / f"trajectories_{chunk_index:04d}.npy"
        if metadata_path.exists() and trajectory_path.exists():
            continue
        start_plant = chunk_index * config.chunk_size_plants
        end_plant = min(config.n_plants, start_plant + config.chunk_size_plants)
        frame, trajectories = _generate_chunk(config, time_grid, start_plant, end_plant)
        frame.to_csv(metadata_path, index=False)
        np.save(trajectory_path, trajectories)

    return consolidate_dataset(config)


def consolidate_dataset(config: DatasetConfig) -> Path:
    dataset_dir = ensure_dir(resolve_path(config.dataset_dir()))
    chunks_dir = dataset_dir / "chunks"
    metadata_paths = sorted(chunks_dir.glob("metadata_*.csv"))
    trajectory_paths = sorted(chunks_dir.glob("trajectories_*.npy"))
    if not metadata_paths or not trajectory_paths:
        raise FileNotFoundError("No dataset chunks found. Run generate_dataset first.")
    if len(metadata_paths) != len(trajectory_paths):
        raise RuntimeError("Chunk metadata and trajectory files are out of sync.")

    samples = pd.concat([pd.read_csv(path) for path in metadata_paths], ignore_index=True)
    trajectories = np.concatenate(
        [np.load(path) for path in trajectory_paths],
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

    samples.to_csv(dataset_dir / "samples.csv.gz", index=False, compression="gzip")
    np.save(dataset_dir / "trajectories.npy", trajectories)

    metadata = _dataset_metadata(config, samples, trajectories)
    dump_json(metadata, dataset_dir / "metadata.json")
    return dataset_dir


def load_dataset(dataset_dir: str | Path) -> DatasetBundle:
    resolved = resolve_path(dataset_dir)
    samples = pd.read_csv(resolved / "samples.csv.gz")
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
) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, Any]] = []
    trajectories: list[np.ndarray] = []

    for plant_id in range(start_plant, end_plant):
        rng = np.random.default_rng(config.seed + (plant_id * 9_973))
        plant = sample_plant(rng, plant_id=plant_id, families=config.families)
        for controller_index in range(config.controllers_per_plant):
            kp, ki, kd = sample_pid_gains(
                rng,
                plant,
                tuple(config.kp_multiplier_range),
                tuple(config.ki_multiplier_range),
                tuple(config.kd_multiplier_range),
            )
            result = simulate_closed_loop(
                plant=plant,
                kp=kp,
                ki=ki,
                kd=kd,
                tau_d=config.derivative_filter_tau,
                time_grid=time_grid,
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
                "stable": bool(result.stable),
                "stability_margin": result.stability_margin,
                "failure_reason": result.reason or "",
            }

            padded_num = plant.padded_numerator()
            padded_den = plant.padded_denominator()
            for index, value in enumerate(padded_num):
                row[f"num_{index}"] = float(value)
            for index, value in enumerate(padded_den):
                row[f"den_{index}"] = float(value)

            if result.stable and result.trajectory is not None:
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

    return pd.DataFrame(rows), np.vstack(trajectories)


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
    return {
        "name": config.name,
        "n_samples": int(samples.shape[0]),
        "n_plants": int(samples["plant_id"].nunique()),
        "n_time_steps": int(trajectories.shape[1]),
        "stable_fraction_pct": round(float(samples["stable"].mean() * 100.0), 2),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "split_stable_fraction_pct": {key: float(value) for key, value in stable_counts.items()},
        "family_counts": {key: int(value) for key, value in family_counts.items()},
        "ood_families": list(config.ood_families),
        "families": list(config.families),
        "controllers_per_plant": config.controllers_per_plant,
        "time_horizon": config.t_final,
    }
