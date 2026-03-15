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
    max_peak_control_effort: float = 250.0
    max_unstable_fraction_abort: float = 0.5
    write_consolidated_outputs: bool = True
    consolidation_sample_limit: int = 1_000_000

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


def load_config[T](path: str | Path, cls: type[T]) -> T:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return cls(**payload)


def save_config_snapshot(config: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
