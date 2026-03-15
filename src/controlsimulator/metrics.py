from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(slots=True)
class ResponseMetrics:
    overshoot_pct: float
    rise_time: float
    settling_time: float
    steady_state_error: float
    peak_control_effort: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


def extract_response_metrics(
    time_grid: np.ndarray,
    trajectory: np.ndarray,
    target: float = 1.0,
    rise_lower: float = 0.1,
    rise_upper: float = 0.9,
    settling_band: float = 0.02,
    peak_control_effort: float | None = None,
) -> ResponseMetrics:
    if time_grid.shape[0] != trajectory.shape[0]:
        raise ValueError("time_grid and trajectory must have the same length.")

    final_value = float(trajectory[-1])
    steady_state_error = abs(target - final_value)
    overshoot_pct = max(
        0.0,
        ((float(np.max(trajectory)) - target) / max(abs(target), 1e-8)) * 100.0,
    )

    lower_level = rise_lower * target
    upper_level = rise_upper * target

    lower_hits = np.where(trajectory >= lower_level)[0]
    upper_hits = np.where(trajectory >= upper_level)[0]
    if lower_hits.size and upper_hits.size:
        rise_time = float(time_grid[upper_hits[0]] - time_grid[lower_hits[0]])
    else:
        rise_time = float("nan")

    band = settling_band * max(abs(target), 1e-8)
    outside = np.where(np.abs(trajectory - target) > band)[0]
    if outside.size == 0:
        settling_time = 0.0
    elif outside[-1] == time_grid.shape[0] - 1:
        settling_time = float("nan")
    else:
        settling_time = float(time_grid[outside[-1] + 1])

    return ResponseMetrics(
        overshoot_pct=overshoot_pct,
        rise_time=rise_time,
        settling_time=settling_time,
        steady_state_error=steady_state_error,
        peak_control_effort=peak_control_effort,
    )


def trajectory_rmse(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def trajectory_mae(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - targets)))


def metric_mae(reference: np.ndarray, predicted: np.ndarray) -> float:
    mask = np.isfinite(reference) & np.isfinite(predicted)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(predicted[mask] - reference[mask])))
