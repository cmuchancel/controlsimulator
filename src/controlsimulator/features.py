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


def build_feature_table(samples: pd.DataFrame) -> pd.DataFrame:
    feature_frame = samples.copy()
    feature_frame["log10_kp"] = np.log10(np.clip(feature_frame["kp"], 1e-8, None))
    feature_frame["log10_ki"] = np.log10(np.clip(feature_frame["ki"], 1e-8, None))
    feature_frame["log10_kd"] = np.log10(np.clip(feature_frame["kd"], 1e-8, None))
    return feature_frame[FEATURE_COLUMNS]


def feature_matrix(samples: pd.DataFrame) -> np.ndarray:
    return build_feature_table(samples).to_numpy(dtype=np.float32)
