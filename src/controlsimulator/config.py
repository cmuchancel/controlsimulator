from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(slots=True)
class DatasetConfig:
    name: str
    seed: int = 42
    output_dir: str = "artifacts/datasets"
    n_plants: int = 240
    controllers_per_plant: int = 12
    num_workers: int = 1
    families: list[str] = field(
        default_factory=lambda: [
            "first_order",
            "second_order",
            "underdamped_second_order",
            "overdamped_second_order",
            "lightly_damped_second_order",
            "highly_resonant_second_order",
            "third_order",
            "third_order_real_poles",
            "third_order_mixed_real_complex",
            "weakly_resonant_third_order",
            "fourth_order_real",
            "fourth_order_mixed_complex",
            "two_mode_resonant",
            "near_integrator",
            "slow_dynamics_family",
            "fast_dynamics_family",
        ]
    )
    family_sampling_weights: dict[str, float] | None = None
    ood_families: list[str] = field(default_factory=lambda: ["lightly_damped_second_order"])
    t_final: float = 8.0
    n_time_steps: int = 200
    derivative_filter_tau: float = 0.05
    chunk_size_plants: int = 40
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    kp_multiplier_range: list[float] = field(default_factory=lambda: [0.02, 50.0])
    ki_multiplier_range: list[float] = field(default_factory=lambda: [0.01, 80.0])
    kd_multiplier_range: list[float] = field(default_factory=lambda: [0.001, 25.0])
    controller_mode_weights: dict[str, float] | None = None
    wide_sampling_fraction: float = 0.5
    boundary_sampling_fraction: float | None = None
    oscillatory_sampling_fraction: float = 0.0
    boundary_search_steps: int = 6
    boundary_mix_std: float = 0.12
    boundary_jitter_std: float = 0.08
    oscillatory_candidate_count: int = 14
    oscillatory_boundary_bias: float = 0.82
    oscillatory_boundary_std: float = 0.1
    oscillatory_jitter_std: float = 0.06
    max_abs_trajectory: float = 12.0
    trajectory_storage_clip_abs: float = 100.0
    trajectory_storage_dtype: str = "float32"
    max_peak_control_effort: float = 250.0
    max_unstable_fraction_abort: float = 0.5
    write_consolidated_outputs: bool = True
    consolidation_sample_limit: int = 1_000_000
    export_dataset_layout: bool = False
    near_instability_margin: float = 0.05
    oscillatory_frequency_threshold_hz: float = 0.2
    unstable_response_target: int | None = None
    oscillatory_response_target: int | None = None
    near_instability_response_target: int | None = None
    quota_resample_batch_plants: int = 2_000
    quota_max_rounds: int = 32

    def __post_init__(self) -> None:
        if self.boundary_sampling_fraction is None:
            self.boundary_sampling_fraction = (
                1.0 - self.wide_sampling_fraction - self.oscillatory_sampling_fraction
            )
        total = (
            self.wide_sampling_fraction
            + float(self.boundary_sampling_fraction)
            + self.oscillatory_sampling_fraction
        )
        if total <= 0.0:
            raise ValueError("At least one controller sampling mode must have positive mass.")
        if not np.isclose(total, 1.0):
            self.wide_sampling_fraction /= total
            self.boundary_sampling_fraction = float(self.boundary_sampling_fraction) / total
            self.oscillatory_sampling_fraction /= total

    def dataset_dir(self) -> Path:
        return Path(self.output_dir) / self.name


@dataclass(slots=True)
class TrainingConfig:
    name: str
    dataset_dir: str
    output_dir: str = "artifacts/runs"
    seed: int = 42
    device: str = "auto"
    batch_size: int = 256
    epochs: int = 80
    patience: int = 12
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256, 256])
    classifier_hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.1
    classifier_dropout: float = 0.1
    feature_set: str = "full"
    activation: str = "gelu"
    classifier_activation: str | None = None

    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


@dataclass(slots=True)
class EvaluationConfig:
    name: str
    dataset_dir: str
    run_dir: str
    output_dir: str = "reports/evaluations"
    benchmark_batch_size: int = 128
    benchmark_single_repeats: int = 250
    benchmark_batch_repeats: int = 80
    inference_batch_size: int = 4096
    knn_neighbors: int = 1
    knn_train_cap: int = 100_000
    knn_seed: int = 42

    def report_dir(self) -> Path:
        return Path(self.output_dir) / self.name


@dataclass(slots=True)
class PublicationEvaluationConfig:
    name: str
    dataset_dir: str
    run_dir: str
    output_tables_dir: str = "artifacts/eval_tables"
    output_plots_dir: str = "artifacts/eval_plots"
    speed_benchmark_path: str = "artifacts/speed_benchmark.json"
    seed: int = 42
    device: str = "auto"
    inference_batch_size: int = 4096
    speed_benchmark_samples: int = 10_000
    property_plot_sample_cap: int = 8_000
    case_candidate_cap: int = 128
    pid_demo_split: str = "ood_test"
    pid_demo_family: str = "campaign_third_order_ood_lightly_damped"
    pid_demo_gradient_steps: int = 80
    pid_demo_learning_rate: float = 0.05
    pid_demo_grid_kp_points: int = 8
    pid_demo_grid_ki_points: int = 8
    pid_demo_grid_kd_points: int = 6

    def tables_dir(self) -> Path:
        return Path(self.output_tables_dir)

    def plots_dir(self) -> Path:
        return Path(self.output_plots_dir)

    def speed_benchmark_file(self) -> Path:
        return Path(self.speed_benchmark_path)


@dataclass(slots=True)
class PIDOptimizationComparisonConfig:
    name: str
    dataset_dir: str
    run_dir: str
    output_tables_dir: str = "artifacts/eval_tables"
    output_plots_dir: str = "artifacts/eval_plots"
    summary_json_path: str = "artifacts/eval_tables/pid_optimization_comparison_summary.json"
    seed: int = 42
    device: str = "auto"
    n_plants: int = 100
    source_splits: list[str] = field(default_factory=lambda: ["test", "ood_test"])
    exclude_families: list[str] = field(
        default_factory=lambda: ["campaign_third_order_unstable"]
    )
    grid_kp_points: int = 6
    grid_ki_points: int = 6
    grid_kd_points: int = 5
    search_log10_span_kp: float = 1.1
    search_log10_span_ki: float = 1.3
    search_log10_span_kd: float = 1.3
    bayes_initial_points: int = 8
    bayes_iterations: int = 24
    bayes_candidate_pool: int = 768
    surrogate_steps: int = 60
    surrogate_learning_rate: float = 0.05

    def tables_dir(self) -> Path:
        return Path(self.output_tables_dir)

    def plots_dir(self) -> Path:
        return Path(self.output_plots_dir)

    def summary_file(self) -> Path:
        return Path(self.summary_json_path)


@dataclass(slots=True)
class SurrogateWarmStartBOConfig:
    name: str
    dataset_dir: str
    run_dir: str
    output_tables_dir: str = "artifacts/eval_tables"
    output_plots_dir: str = "artifacts/eval_plots"
    summary_json_path: str = "artifacts/eval_tables/pid_surrogate_bo_summary.json"
    seed: int = 42
    device: str = "auto"
    n_plants: int = 100
    source_splits: list[str] = field(default_factory=lambda: ["test", "ood_test"])
    exclude_families: list[str] = field(
        default_factory=lambda: ["campaign_third_order_unstable"]
    )
    search_log10_span_kp: float = 1.1
    search_log10_span_ki: float = 1.3
    search_log10_span_kd: float = 1.3
    bayes_initial_points: int = 8
    bayes_iterations: int = 24
    bayes_candidate_pool: int = 768
    surrogate_steps: int = 60
    surrogate_learning_rate: float = 0.05
    surrogate_bo_iterations: int = 10
    surrogate_bo_lower_factor: float = 0.5
    surrogate_bo_upper_factor: float = 1.5
    surrogate_bo_candidate_pool: int = 512

    def tables_dir(self) -> Path:
        return Path(self.output_tables_dir)

    def plots_dir(self) -> Path:
        return Path(self.output_plots_dir)

    def summary_file(self) -> Path:
        return Path(self.summary_json_path)


def load_config[T](path: str | Path, cls: type[T]) -> T:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return cls(**payload)


def save_config_snapshot(config: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
