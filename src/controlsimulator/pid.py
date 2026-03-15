from __future__ import annotations

import numpy as np


def pid_controller_coefficients(
    kp: float,
    ki: float,
    kd: float,
    tau_d: float,
) -> tuple[np.ndarray, np.ndarray]:
    if tau_d < 0:
        raise ValueError("tau_d must be non-negative.")
    if tau_d == 0:
        numerator = np.array([kd, kp, ki], dtype=np.float64)
        denominator = np.array([1.0, 0.0], dtype=np.float64)
        return numerator, denominator
    numerator = np.array(
        [
            (kp * tau_d) + kd,
            kp + (ki * tau_d),
            ki,
        ],
        dtype=np.float64,
    )
    denominator = np.array([tau_d, 1.0, 0.0], dtype=np.float64)
    return numerator, denominator
