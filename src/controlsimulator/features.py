from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from controlsimulator.plants import MAX_DEN_ORDER, MAX_NUM_ORDER

NUMERATOR_COLUMNS = [f"num_{index}" for index in range(MAX_NUM_ORDER + 1)]
DENOMINATOR_COLUMNS = [f"den_{index}" for index in range(MAX_DEN_ORDER + 1)]
PLANT_FEATURE_COLUMNS = [
    *NUMERATOR_COLUMNS,
    *DENOMINATOR_COLUMNS,
    "dc_gain",
    "dominant_pole_mag",
    "mean_pole_mag",
    "plant_order",
    "plant_min_damping_ratio",
    "plant_max_oscillation_hz",
    "plant_pole_spread_log10",
    "plant_has_complex_poles",
]
GAIN_FEATURE_COLUMNS = ["kp", "ki", "kd", "log10_kp", "log10_ki", "log10_kd"]
FEATURE_COLUMNS = [*PLANT_FEATURE_COLUMNS, *GAIN_FEATURE_COLUMNS]


@dataclass(slots=True)
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> Standardizer:
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    @classmethod
    def from_running(cls, stats: RunningStatistics) -> Standardizer:
        return stats.to_standardizer()

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return (values * self.std) + self.mean

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> Standardizer:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


@dataclass(slots=True)
class RunningStatistics:
    total: np.ndarray | None = None
    total_squared: np.ndarray | None = None
    count: int = 0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        batch_sum = values.sum(axis=0)
        batch_squared = np.square(values).sum(axis=0)
        if self.total is None or self.total_squared is None:
            self.total = batch_sum
            self.total_squared = batch_squared
        else:
            self.total += batch_sum
            self.total_squared += batch_squared
        self.count += int(values.shape[0])

    def to_standardizer(self) -> Standardizer:
        if self.count == 0 or self.total is None or self.total_squared is None:
            raise ValueError("Cannot build a standardizer from empty running statistics.")
        mean = self.total / self.count
        variance = (self.total_squared / self.count) - np.square(mean)
        variance = np.maximum(variance, 1e-12)
        std = np.sqrt(variance)
        std = np.where(std < 1e-6, 1.0, std)
        return Standardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def _ensure_feature_columns(samples: pd.DataFrame) -> pd.DataFrame:
    feature_frame = samples.copy()
    defaults = {
        "plant_min_damping_ratio": 1.0,
        "plant_max_oscillation_hz": 0.0,
        "plant_pole_spread_log10": 0.0,
        "plant_has_complex_poles": 0.0,
    }
    for column in FEATURE_COLUMNS:
        if column in feature_frame.columns:
            continue
        feature_frame[column] = defaults.get(column, 0.0)
    return feature_frame


def build_feature_table(samples: pd.DataFrame) -> pd.DataFrame:
    feature_frame = _ensure_feature_columns(samples)
    feature_frame["log10_kp"] = np.log10(np.clip(feature_frame["kp"], 1e-8, None))
    feature_frame["log10_ki"] = np.log10(np.clip(feature_frame["ki"], 1e-8, None))
    feature_frame["log10_kd"] = np.log10(np.clip(feature_frame["kd"], 1e-8, None))
    feature_frame["plant_has_complex_poles"] = feature_frame["plant_has_complex_poles"].astype(
        float
    )
    return feature_frame[FEATURE_COLUMNS]


def feature_matrix(samples: pd.DataFrame) -> np.ndarray:
    return build_feature_table(samples).to_numpy(dtype=np.float32)
