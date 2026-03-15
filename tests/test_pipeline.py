from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from controlsimulator.config import DatasetConfig, EvaluationConfig, TrainingConfig
from controlsimulator.dataset import generate_dataset, load_dataset
from controlsimulator.evaluate import evaluate_models
from controlsimulator.train import train_models


def test_smoke_pipeline_runs(tmp_path: Path) -> None:
    dataset_config = DatasetConfig(
        name="tiny_smoke",
        seed=5,
        output_dir=str(tmp_path / "datasets"),
        n_plants=40,
        controllers_per_plant=3,
        num_workers=1,
        chunk_size_plants=10,
        t_final=6.0,
        n_time_steps=80,
    )
    dataset_dir = generate_dataset(dataset_config)

    training_config = TrainingConfig(
        name="tiny_run",
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "runs"),
        seed=5,
        device="cpu",
        batch_size=32,
        epochs=2,
        patience=1,
        hidden_sizes=[32, 32],
        classifier_hidden_sizes=[16, 16],
        dropout=0.0,
        classifier_dropout=0.0,
    )
    run_dir = train_models(training_config)

    evaluation_config = EvaluationConfig(
        name="tiny_run",
        dataset_dir=str(dataset_dir),
        run_dir=str(run_dir),
        output_dir=str(tmp_path / "reports"),
        benchmark_batch_size=16,
        benchmark_single_repeats=5,
        benchmark_batch_repeats=2,
    )
    report_dir = evaluate_models(evaluation_config)

    assert (dataset_dir / "samples.parquet").exists()
    assert (dataset_dir / "trajectories.npz").exists()
    assert (run_dir / "regressor.pt").exists()
    assert (run_dir / "classifier.pt").exists()
    assert (report_dir / "evaluation_summary.json").exists()


def test_dataset_generation_is_deterministic_across_workers(tmp_path: Path) -> None:
    base_kwargs = dict(
        seed=17,
        n_plants=18,
        controllers_per_plant=4,
        chunk_size_plants=6,
        t_final=5.0,
        n_time_steps=60,
        families=["first_order", "second_order", "third_order_real_poles"],
    )
    config_single = DatasetConfig(
        name="deterministic_single",
        output_dir=str(tmp_path / "datasets"),
        num_workers=1,
        **base_kwargs,
    )
    config_parallel = DatasetConfig(
        name="deterministic_parallel",
        output_dir=str(tmp_path / "datasets"),
        num_workers=2,
        **base_kwargs,
    )

    single_dir = generate_dataset(config_single)
    parallel_dir = generate_dataset(config_parallel)

    single_bundle = load_dataset(single_dir)
    parallel_bundle = load_dataset(parallel_dir)
    single_samples = single_bundle.samples.sort_values("sample_id").reset_index(drop=True)
    parallel_samples = parallel_bundle.samples.sort_values("sample_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(single_samples, parallel_samples, check_like=True)
    assert np.array_equal(single_bundle.trajectories, parallel_bundle.trajectories, equal_nan=True)


def test_dataset_directory_rejects_config_mismatch(tmp_path: Path) -> None:
    config = DatasetConfig(
        name="config_mismatch",
        output_dir=str(tmp_path / "datasets"),
        n_plants=12,
        controllers_per_plant=3,
        num_workers=1,
        chunk_size_plants=6,
    )
    generate_dataset(config)

    changed = DatasetConfig(
        name="config_mismatch",
        output_dir=str(tmp_path / "datasets"),
        n_plants=12,
        controllers_per_plant=5,
        num_workers=1,
        chunk_size_plants=6,
    )
    with pytest.raises(RuntimeError):
        generate_dataset(changed)
