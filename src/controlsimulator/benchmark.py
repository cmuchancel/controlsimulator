from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from controlsimulator.config import EvaluationConfig
from controlsimulator.dataset import load_dataset
from controlsimulator.evaluate import (
    load_models,
    predict_stability_probabilities,
    predict_trajectories,
)
from controlsimulator.features import feature_matrix
from controlsimulator.plants import plant_from_sample_row
from controlsimulator.simulate import simulate_closed_loop
from controlsimulator.utils import dump_json, ensure_dir, resolve_path


def benchmark_models(config: EvaluationConfig) -> Path:
    dataset = load_dataset(config.dataset_dir)
    models = load_models(config.run_dir)
    report_dir = ensure_dir(resolve_path(config.report_dir()))

    features = feature_matrix(dataset.samples)
    scaled_features = models.feature_scaler.transform(features).astype(np.float32)
    stable_test = dataset.samples[
        (dataset.samples["split"] == "test") & (dataset.samples["stable"])
    ].copy()
    if stable_test.empty:
        raise RuntimeError("No stable test samples available for benchmarking.")

    single_index = int(stable_test.index[0])
    batch_indices = stable_test.index.to_numpy(dtype=int)[: config.benchmark_batch_size]

    single_sample = dataset.samples.loc[single_index]
    single_feature = scaled_features[single_index : single_index + 1]
    batch_features = scaled_features[batch_indices]

    single_simulation = _benchmark_repeat(
        lambda: simulate_closed_loop(
            plant=plant_from_sample_row(single_sample),
            kp=float(single_sample["kp"]),
            ki=float(single_sample["ki"]),
            kd=float(single_sample["kd"]),
            tau_d=float(single_sample["tau_d"]),
            time_grid=dataset.time_grid,
        ),
        config.benchmark_single_repeats,
    )
    single_surrogate = _benchmark_repeat(
        lambda: _full_surrogate_pass(models, single_feature),
        config.benchmark_single_repeats,
    )

    batch_simulation = _benchmark_repeat(
        lambda: [
            simulate_closed_loop(
                plant=plant_from_sample_row(dataset.samples.loc[index]),
                kp=float(dataset.samples.loc[index, "kp"]),
                ki=float(dataset.samples.loc[index, "ki"]),
                kd=float(dataset.samples.loc[index, "kd"]),
                tau_d=float(dataset.samples.loc[index, "tau_d"]),
                time_grid=dataset.time_grid,
            )
            for index in batch_indices
        ],
        config.benchmark_batch_repeats,
    )
    batch_surrogate = _benchmark_repeat(
        lambda: _full_surrogate_pass(models, batch_features),
        config.benchmark_batch_repeats,
    )

    summary = {
        "single_simulation_seconds": single_simulation,
        "single_surrogate_seconds": single_surrogate,
        "single_speedup_x": single_simulation / max(single_surrogate, 1e-9),
        "batch_size": int(batch_features.shape[0]),
        "batch_simulation_seconds": batch_simulation,
        "batch_surrogate_seconds": batch_surrogate,
        "batch_speedup_x": batch_simulation / max(batch_surrogate, 1e-9),
        "batch_simulation_per_sample_ms": (batch_simulation / batch_features.shape[0]) * 1000.0,
        "batch_surrogate_per_sample_ms": (batch_surrogate / batch_features.shape[0]) * 1000.0,
    }
    dump_json(summary, report_dir / "benchmark_summary.json")
    _write_benchmark_markdown(summary, report_dir / "benchmark_summary.md")
    return report_dir


def _full_surrogate_pass(models: Any, scaled_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stability_probabilities = predict_stability_probabilities(models.classifier, scaled_features)
    trajectories = predict_trajectories(models.regressor, scaled_features, models.trajectory_scaler)
    return stability_probabilities, trajectories


def _benchmark_repeat(function: Any, repeats: int) -> float:
    function()
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        function()
        durations.append(perf_counter() - start)
    return float(np.median(durations))


def _write_benchmark_markdown(summary: dict[str, float], path: str | Path) -> None:
    lines = [
        "# Benchmark Summary",
        "",
        f"- single simulation: {summary['single_simulation_seconds']:.6f} s",
        f"- single surrogate: {summary['single_surrogate_seconds']:.6f} s",
        f"- single speedup: {summary['single_speedup_x']:.2f}x",
        f"- batch size: {int(summary['batch_size'])}",
        f"- batch simulation: {summary['batch_simulation_seconds']:.6f} s",
        f"- batch surrogate: {summary['batch_surrogate_seconds']:.6f} s",
        f"- batch speedup: {summary['batch_speedup_x']:.2f}x",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
