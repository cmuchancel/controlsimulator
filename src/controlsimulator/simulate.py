from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import signal

from controlsimulator.pid import pid_controller_coefficients
from controlsimulator.plants import Plant


@dataclass(slots=True)
class SimulationResult:
    stable: bool
    stability_margin: float
    trajectory: np.ndarray | None
    control_effort: np.ndarray | None
    peak_control_effort: float | None
    reason: str | None = None


def multiply_transfer_functions(
    numerator_a: np.ndarray,
    denominator_a: np.ndarray,
    numerator_b: np.ndarray,
    denominator_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return np.convolve(numerator_a, numerator_b), np.convolve(denominator_a, denominator_b)


def _add_polynomials(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    width = max(left.shape[0], right.shape[0])
    padded_left = np.pad(left, (width - left.shape[0], 0))
    padded_right = np.pad(right, (width - right.shape[0], 0))
    return padded_left + padded_right


def closed_loop_transfer_function(
    plant: Plant,
    kp: float,
    ki: float,
    kd: float,
    tau_d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    controller_num, controller_den = pid_controller_coefficients(kp, ki, kd, tau_d)
    open_loop_num, open_loop_den = multiply_transfer_functions(
        controller_num,
        controller_den,
        plant.numerator,
        plant.denominator,
    )
    closed_loop_num = open_loop_num
    closed_loop_den = _add_polynomials(open_loop_den, open_loop_num)
    return controller_num, controller_den, closed_loop_num, closed_loop_den


def stability_margin(denominator: np.ndarray) -> float:
    roots = np.roots(denominator)
    return float(np.max(np.real(roots)))


def is_stable(denominator: np.ndarray, tolerance: float = -1e-6) -> bool:
    return stability_margin(denominator) < tolerance


def closed_loop_stability_margin(
    plant: Plant,
    kp: float,
    ki: float,
    kd: float,
    tau_d: float,
) -> float:
    *_, closed_loop_den = closed_loop_transfer_function(plant, kp, ki, kd, tau_d)
    return stability_margin(closed_loop_den)


def closed_loop_is_stable(
    plant: Plant,
    kp: float,
    ki: float,
    kd: float,
    tau_d: float,
    tolerance: float = -1e-6,
) -> bool:
    return closed_loop_stability_margin(plant, kp, ki, kd, tau_d) < tolerance


def _step_response(
    numerator: np.ndarray,
    denominator: np.ndarray,
    time_grid: np.ndarray,
) -> np.ndarray:
    system = signal.TransferFunction(numerator, denominator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, response = signal.step(system, T=time_grid)
    return np.asarray(response, dtype=np.float64)


def simulate_closed_loop(
    plant: Plant,
    kp: float,
    ki: float,
    kd: float,
    tau_d: float,
    time_grid: np.ndarray,
) -> SimulationResult:
    (
        controller_num,
        controller_den,
        closed_loop_num,
        closed_loop_den,
    ) = closed_loop_transfer_function(plant, kp, ki, kd, tau_d)
    margin = stability_margin(closed_loop_den)
    if margin >= -1e-6:
        return SimulationResult(
            stable=False,
            stability_margin=margin,
            trajectory=None,
            control_effort=None,
            peak_control_effort=None,
            reason="unstable_closed_loop",
        )

    try:
        trajectory = _step_response(closed_loop_num, closed_loop_den, time_grid)
        control_num = np.convolve(controller_num, plant.denominator)
        control_effort = _step_response(control_num, closed_loop_den, time_grid)
    except Exception as error:  # pragma: no cover - defensive boundary
        return SimulationResult(
            stable=False,
            stability_margin=margin,
            trajectory=None,
            control_effort=None,
            peak_control_effort=None,
            reason=f"simulation_error:{type(error).__name__}",
        )

    if not np.all(np.isfinite(trajectory)) or not np.all(np.isfinite(control_effort)):
        return SimulationResult(
            stable=False,
            stability_margin=margin,
            trajectory=None,
            control_effort=None,
            peak_control_effort=None,
            reason="non_finite_response",
        )

    peak_control_effort = float(np.max(np.abs(control_effort)))
    return SimulationResult(
        stable=True,
        stability_margin=margin,
        trajectory=trajectory.astype(np.float32),
        control_effort=control_effort.astype(np.float32),
        peak_control_effort=peak_control_effort,
        reason=None,
    )
