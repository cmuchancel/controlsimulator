from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DatasetConfig:
    name: str
    seed: int = 42
    output_dir: str = "artifacts/datasets"
    n_plants: int = 240
    controllers_per_plant: int = 12
    families: list[str] = field(
        default_factory=lambda: [
            "first_order",
            "second_order",
            "third_order",
            "lightly_damped_second_order",
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

    def report_dir(self) -> Path:
        return Path(self.output_dir) / self.name


def load_config[T](path: str | Path, cls: type[T]) -> T:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return cls(**payload)


def save_config_snapshot(config: Any, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
