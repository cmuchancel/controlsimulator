from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from tqdm import tqdm

from controlsimulator.config import EvaluationConfig
from controlsimulator.dataset import dataset_time_grid, iter_dataset_chunks
from controlsimulator.evaluate import (
    load_models,
    predict_stability_probabilities,
    predict_trajectories,
)
from controlsimulator.features import GAIN_FEATURE_COLUMNS, PLANT_FEATURE_COLUMNS, feature_matrix
from controlsimulator.plants import plant_from_sample_row
from controlsimulator.simulate import simulate_closed_loop
from controlsimulator.utils import dump_json, ensure_dir, resolve_path

RAW_FEATURE_COLUMNS = [
    *PLANT_FEATURE_COLUMNS,
    *[column for column in GAIN_FEATURE_COLUMNS if not column.startswith("log10_")],
]


def _benchmark_columns(feature_columns: list[str]) -> list[str]:
    raw_columns = [column for column in feature_columns if not column.startswith("log10_")]
    return [
        *raw_columns,
        "sample_id",
        "plant_id",
        "tau_d",
        "kp",
        "ki",
        "kd",
        "stable",
        "split",
    ]


def _log_benchmark_stage(message: str) -> None:
    print(message, flush=True)


def benchmark_models(config: EvaluationConfig) -> Path:
    dataset_dir = resolve_path(config.dataset_dir)
    time_grid = dataset_time_grid(dataset_dir)
    models = load_models(config.run_dir)
    report_dir = ensure_dir(resolve_path(config.report_dir()))

    _log_benchmark_stage(f"[benchmark] load {config.name}")
    stable_test_frame = None
    stable_test_trajectories = None
    for frame, trajectories in iter_dataset_chunks(
        dataset_dir,
        include_trajectories=True,
        splits={"test"},
        stable_only=True,
        columns=_benchmark_columns(models.feature_columns),
        progress_desc=f"benchmark-load:{config.name}",
    ):
        stable_test_frame = frame
        stable_test_trajectories = trajectories
        break
    if stable_test_frame is None or stable_test_frame.empty or stable_test_trajectories is None:
        raise RuntimeError("No stable test samples available for benchmarking.")

    batch_frame = stable_test_frame.iloc[: config.benchmark_batch_size].copy()
    batch_features = models.feature_scaler.transform(
        feature_matrix(batch_frame, feature_columns=models.feature_columns)
    ).astype(np.float32)

    single_sample = batch_frame.iloc[0]
    single_feature = batch_features[:1]

    _log_benchmark_stage(f"[benchmark] single-sim {config.name}")
    single_simulation = _benchmark_repeat(
        lambda: simulate_closed_loop(
            plant=plant_from_sample_row(single_sample),
            kp=float(single_sample["kp"]),
            ki=float(single_sample["ki"]),
            kd=float(single_sample["kd"]),
            tau_d=float(single_sample["tau_d"]),
            time_grid=time_grid,
        ),
        config.benchmark_single_repeats,
        desc=f"benchmark-single-sim:{config.name}",
    )
    _log_benchmark_stage(f"[benchmark] single-surrogate {config.name}")
    single_surrogate = _benchmark_repeat(
        lambda: _full_surrogate_pass(models, single_feature, config.inference_batch_size),
        config.benchmark_single_repeats,
        desc=f"benchmark-single-surrogate:{config.name}",
    )

    _log_benchmark_stage(f"[benchmark] batch-sim {config.name}")
    batch_simulation = _benchmark_repeat(
        lambda: [
            simulate_closed_loop(
                plant=plant_from_sample_row(batch_frame.iloc[index]),
                kp=float(batch_frame.iloc[index]["kp"]),
                ki=float(batch_frame.iloc[index]["ki"]),
                kd=float(batch_frame.iloc[index]["kd"]),
                tau_d=float(batch_frame.iloc[index]["tau_d"]),
                time_grid=time_grid,
            )
            for index in range(batch_frame.shape[0])
        ],
        config.benchmark_batch_repeats,
        desc=f"benchmark-batch-sim:{config.name}",
    )
    _log_benchmark_stage(f"[benchmark] batch-surrogate {config.name}")
    batch_surrogate = _benchmark_repeat(
        lambda: _full_surrogate_pass(models, batch_features, config.inference_batch_size),
        config.benchmark_batch_repeats,
        desc=f"benchmark-batch-surrogate:{config.name}",
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
    _log_benchmark_stage(f"[benchmark] complete {config.name}")
    return report_dir


def _full_surrogate_pass(
    models: Any,
    scaled_features: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    stability_probabilities = predict_stability_probabilities(
        models.classifier,
        scaled_features,
        batch_size=batch_size,
    )
    trajectories = predict_trajectories(
        models.regressor,
        scaled_features,
        models.trajectory_scaler,
        batch_size=batch_size,
    )
    return stability_probabilities, trajectories


def _benchmark_repeat(function: Any, repeats: int, *, desc: str) -> float:
    function()
    durations = []
    for _ in tqdm(range(repeats), desc=desc, unit="run"):
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
