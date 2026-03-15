from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from controlsimulator.campaign import run_campaign
from controlsimulator.config import (
    DatasetConfig,
    EvaluationConfig,
    PIDOptimizationComparisonConfig,
    PublicationEvaluationConfig,
    SurrogateWarmStartBOConfig,
    TrainingConfig,
)
from controlsimulator.dataset import generate_dataset, load_dataset
from controlsimulator.evaluate import evaluate_models
from controlsimulator.pid_optimization_compare import run_pid_optimization_comparison
from controlsimulator.pid_surrogate_bo import run_surrogate_warm_start_bo_experiment
from controlsimulator.publication_eval import run_publication_evaluation
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


def test_campaign_pipeline_exports_expected_artifacts(tmp_path: Path) -> None:
    dataset_config = DatasetConfig(
        name="campaign_tiny",
        seed=13,
        output_dir=str(tmp_path / "datasets"),
        n_plants=36,
        controllers_per_plant=4,
        num_workers=1,
        families=[
            "campaign_third_order_stable",
            "campaign_third_order_oscillatory",
            "campaign_third_order_ood_lightly_damped",
            "campaign_third_order_near_instability",
            "campaign_third_order_unstable",
        ],
        family_sampling_weights={
            "campaign_third_order_stable": 0.4,
            "campaign_third_order_oscillatory": 0.2,
            "campaign_third_order_ood_lightly_damped": 0.1,
            "campaign_third_order_near_instability": 0.2,
            "campaign_third_order_unstable": 0.1,
        },
        controller_mode_weights={
            "random": 0.5,
            "aggressive": 0.25,
            "integral_heavy": 0.15,
            "weak": 0.1,
        },
        derivative_filter_tau=0.0,
        chunk_size_plants=12,
        t_final=6.0,
        n_time_steps=121,
        export_dataset_layout=True,
        unstable_response_target=8,
        oscillatory_response_target=10,
        near_instability_response_target=10,
        quota_resample_batch_plants=12,
        quota_max_rounds=8,
        ood_families=["campaign_third_order_ood_lightly_damped"],
        max_unstable_fraction_abort=0.8,
    )
    training_config = TrainingConfig(
        name="campaign_tiny",
        dataset_dir=str(tmp_path / "datasets" / "campaign_tiny"),
        output_dir=str(tmp_path / "runs"),
        seed=13,
        device="cpu",
        batch_size=32,
        epochs=2,
        patience=1,
        hidden_sizes=[32, 32, 32, 32],
        classifier_hidden_sizes=[32, 32, 32, 32],
        dropout=0.0,
        classifier_dropout=0.0,
        feature_set="campaign_core",
        activation="relu",
        classifier_activation="relu",
    )
    evaluation_config = EvaluationConfig(
        name="campaign_tiny",
        dataset_dir=str(tmp_path / "datasets" / "campaign_tiny"),
        run_dir=str(tmp_path / "runs" / "campaign_tiny"),
        output_dir=str(tmp_path / "reports"),
        benchmark_batch_size=8,
        benchmark_single_repeats=2,
        benchmark_batch_repeats=2,
        inference_batch_size=128,
        knn_train_cap=256,
    )

    report_dir = run_campaign(dataset_config, training_config, evaluation_config)
    dataset_dir = tmp_path / "datasets" / "campaign_tiny"

    assert (dataset_dir / "dataset" / "plants.parquet").exists()
    assert (dataset_dir / "dataset" / "controllers.parquet").exists()
    assert (dataset_dir / "dataset" / "metrics.parquet").exists()
    assert (dataset_dir / "dataset" / "labels.parquet").exists()
    assert (dataset_dir / "dataset" / "trajectories.npy").exists()
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "training_curves.png").exists()
    assert (report_dir / "campaign_report.md").exists()


def test_publication_evaluation_runs(tmp_path: Path) -> None:
    dataset_config = DatasetConfig(
        name="publication_tiny",
        seed=21,
        output_dir=str(tmp_path / "datasets"),
        n_plants=30,
        controllers_per_plant=3,
        num_workers=1,
        chunk_size_plants=10,
        t_final=4.0,
        n_time_steps=81,
        families=[
            "campaign_third_order_stable",
            "campaign_third_order_oscillatory",
            "campaign_third_order_ood_lightly_damped",
            "campaign_third_order_near_instability",
            "campaign_third_order_unstable",
        ],
        family_sampling_weights={
            "campaign_third_order_stable": 0.35,
            "campaign_third_order_oscillatory": 0.25,
            "campaign_third_order_ood_lightly_damped": 0.15,
            "campaign_third_order_near_instability": 0.15,
            "campaign_third_order_unstable": 0.10,
        },
        derivative_filter_tau=0.0,
        ood_families=["campaign_third_order_ood_lightly_damped"],
        max_unstable_fraction_abort=0.85,
    )
    dataset_dir = generate_dataset(dataset_config)

    training_config = TrainingConfig(
        name="publication_tiny",
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "runs"),
        seed=21,
        device="cpu",
        batch_size=32,
        epochs=2,
        patience=1,
        hidden_sizes=[32, 32, 32, 32],
        classifier_hidden_sizes=[32, 32, 32, 32],
        dropout=0.0,
        classifier_dropout=0.0,
        feature_set="campaign_core",
        activation="relu",
        classifier_activation="relu",
    )
    run_dir = train_models(training_config)

    publication_config = PublicationEvaluationConfig(
        name="publication_tiny",
        dataset_dir=str(dataset_dir),
        run_dir=str(run_dir),
        output_tables_dir=str(tmp_path / "artifacts" / "eval_tables"),
        output_plots_dir=str(tmp_path / "artifacts" / "eval_plots"),
        speed_benchmark_path=str(tmp_path / "artifacts" / "speed_benchmark.json"),
        seed=21,
        device="cpu",
        inference_batch_size=128,
        speed_benchmark_samples=40,
        property_plot_sample_cap=48,
        case_candidate_cap=16,
        pid_demo_gradient_steps=8,
        pid_demo_grid_kp_points=3,
        pid_demo_grid_ki_points=3,
        pid_demo_grid_kd_points=2,
    )
    output_dir = run_publication_evaluation(publication_config)

    assert (output_dir / "family_ood_analysis.csv").exists()
    assert (output_dir / "pid_optimization_demo.csv").exists()
    assert (output_dir / "publication_eval_summary.json").exists()
    assert (tmp_path / "artifacts" / "eval_plots" / "trajectory_quality_publication.png").exists()
    assert (tmp_path / "artifacts" / "eval_plots" / "error_vs_system_properties.png").exists()
    assert (tmp_path / "artifacts" / "speed_benchmark.json").exists()


def test_pid_optimization_comparison_runs(tmp_path: Path) -> None:
    dataset_config = DatasetConfig(
        name="pid_opt_tiny",
        seed=31,
        output_dir=str(tmp_path / "datasets"),
        n_plants=28,
        controllers_per_plant=3,
        num_workers=1,
        chunk_size_plants=7,
        t_final=4.0,
        n_time_steps=81,
        families=[
            "campaign_third_order_stable",
            "campaign_third_order_oscillatory",
            "campaign_third_order_ood_lightly_damped",
            "campaign_third_order_near_instability",
        ],
        family_sampling_weights={
            "campaign_third_order_stable": 0.35,
            "campaign_third_order_oscillatory": 0.30,
            "campaign_third_order_ood_lightly_damped": 0.20,
            "campaign_third_order_near_instability": 0.15,
        },
        derivative_filter_tau=0.0,
        ood_families=["campaign_third_order_ood_lightly_damped"],
        max_unstable_fraction_abort=0.9,
    )
    dataset_dir = generate_dataset(dataset_config)

    training_config = TrainingConfig(
        name="pid_opt_tiny",
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "runs"),
        seed=31,
        device="cpu",
        batch_size=32,
        epochs=2,
        patience=1,
        hidden_sizes=[32, 32],
        classifier_hidden_sizes=[32, 32],
        dropout=0.0,
        classifier_dropout=0.0,
        feature_set="campaign_core",
        activation="relu",
        classifier_activation="relu",
    )
    run_dir = train_models(training_config)

    comparison_config = PIDOptimizationComparisonConfig(
        name="pid_opt_tiny",
        dataset_dir=str(dataset_dir),
        run_dir=str(run_dir),
        output_tables_dir=str(tmp_path / "tables"),
        output_plots_dir=str(tmp_path / "plots"),
        summary_json_path=str(tmp_path / "tables" / "pid_opt_summary.json"),
        device="cpu",
        n_plants=4,
        source_splits=["train", "val", "test", "ood_test"],
        grid_kp_points=3,
        grid_ki_points=3,
        grid_kd_points=2,
        bayes_initial_points=3,
        bayes_iterations=5,
        bayes_candidate_pool=64,
        surrogate_steps=8,
        surrogate_learning_rate=0.04,
    )
    output_dir = run_pid_optimization_comparison(comparison_config)

    assert (output_dir / "pid_optimization_comparison_results.csv").exists()
    assert (output_dir / "pid_optimization_comparison_summary.csv").exists()
    assert (tmp_path / "plots" / "pid_optimization_cost_boxplot.png").exists()
    assert (tmp_path / "plots" / "pid_optimization_runtime_comparison.png").exists()
    assert (tmp_path / "tables" / "pid_opt_summary.json").exists()


def test_surrogate_warm_start_bo_runs(tmp_path: Path) -> None:
    dataset_config = DatasetConfig(
        name="pid_surrogate_bo_tiny",
        seed=37,
        output_dir=str(tmp_path / "datasets"),
        n_plants=28,
        controllers_per_plant=3,
        num_workers=1,
        chunk_size_plants=7,
        t_final=4.0,
        n_time_steps=81,
        families=[
            "campaign_third_order_stable",
            "campaign_third_order_oscillatory",
            "campaign_third_order_ood_lightly_damped",
            "campaign_third_order_near_instability",
        ],
        family_sampling_weights={
            "campaign_third_order_stable": 0.35,
            "campaign_third_order_oscillatory": 0.30,
            "campaign_third_order_ood_lightly_damped": 0.20,
            "campaign_third_order_near_instability": 0.15,
        },
        derivative_filter_tau=0.0,
        ood_families=["campaign_third_order_ood_lightly_damped"],
        max_unstable_fraction_abort=0.9,
    )
    dataset_dir = generate_dataset(dataset_config)

    training_config = TrainingConfig(
        name="pid_surrogate_bo_tiny",
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "runs"),
        seed=37,
        device="cpu",
        batch_size=32,
        epochs=2,
        patience=1,
        hidden_sizes=[32, 32],
        classifier_hidden_sizes=[32, 32],
        dropout=0.0,
        classifier_dropout=0.0,
        feature_set="campaign_core",
        activation="relu",
        classifier_activation="relu",
    )
    run_dir = train_models(training_config)

    comparison_config = SurrogateWarmStartBOConfig(
        name="pid_surrogate_bo_tiny",
        dataset_dir=str(dataset_dir),
        run_dir=str(run_dir),
        output_tables_dir=str(tmp_path / "tables"),
        output_plots_dir=str(tmp_path / "plots"),
        summary_json_path=str(tmp_path / "tables" / "pid_surrogate_bo_summary.json"),
        device="cpu",
        n_plants=4,
        source_splits=["train", "val", "test", "ood_test"],
        bayes_initial_points=3,
        bayes_iterations=5,
        bayes_candidate_pool=64,
        surrogate_steps=8,
        surrogate_learning_rate=0.04,
        surrogate_bo_iterations=4,
        surrogate_bo_candidate_pool=48,
    )
    output_dir = run_surrogate_warm_start_bo_experiment(comparison_config)

    assert (output_dir / "pid_surrogate_bo_results.csv").exists()
    assert (output_dir / "pid_surrogate_bo_summary.csv").exists()
    assert (tmp_path / "plots" / "pid_surrogate_bo_cost_boxplot.png").exists()
    assert (tmp_path / "plots" / "pid_surrogate_bo_simulation_counts.png").exists()
    assert (tmp_path / "tables" / "pid_surrogate_bo_summary.json").exists()
