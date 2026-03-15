from __future__ import annotations

from pathlib import Path

from controlsimulator.config import DatasetConfig, EvaluationConfig, TrainingConfig
from controlsimulator.dataset import generate_dataset
from controlsimulator.evaluate import evaluate_models
from controlsimulator.train import train_models


def test_smoke_pipeline_runs(tmp_path: Path) -> None:
    dataset_config = DatasetConfig(
        name="tiny_smoke",
        seed=5,
        output_dir=str(tmp_path / "datasets"),
        n_plants=40,
        controllers_per_plant=3,
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

    assert (dataset_dir / "samples.csv.gz").exists()
    assert (run_dir / "regressor.pt").exists()
    assert (run_dir / "classifier.pt").exists()
    assert (report_dir / "evaluation_summary.json").exists()
