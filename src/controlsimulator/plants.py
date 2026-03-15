from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MAX_DEN_ORDER = 3
MAX_NUM_ORDER = 1
FAMILY_CHOICES = [
    "first_order",
    "second_order",
    "third_order",
    "lightly_damped_second_order",
]


@dataclass(slots=True)
class Plant:
    plant_id: int
    family: str
    numerator: np.ndarray
    denominator: np.ndarray
    poles: np.ndarray
    zeros: np.ndarray
    dc_gain: float
    plant_order: int
    dominant_pole_mag: float
    mean_pole_mag: float

    def padded_numerator(self) -> np.ndarray:
        return _pad_coefficients(self.numerator, MAX_NUM_ORDER + 1)

    def padded_denominator(self) -> np.ndarray:
        return _pad_coefficients(self.denominator, MAX_DEN_ORDER + 1)


def _pad_coefficients(coefficients: np.ndarray, length: int) -> np.ndarray:
    if coefficients.shape[0] > length:
        raise ValueError(f"Expected at most {length} coefficients, got {coefficients.shape[0]}")
    padded = np.zeros(length, dtype=np.float64)
    padded[-coefficients.shape[0] :] = coefficients
    return padded


def _make_transfer_function(
    plant_id: int,
    family: str,
    poles: list[complex],
    zeros: list[complex],
    dc_gain: float,
) -> Plant:
    denominator = np.poly(poles).real.astype(np.float64)
    base_numerator = np.poly(zeros).real.astype(np.float64) if zeros else np.array([1.0])
    numerator_scale = dc_gain * denominator[-1] / base_numerator[-1]
    numerator = numerator_scale * base_numerator
    denominator = denominator / denominator[0]
    numerator = numerator / denominator[0]

    poles_array = np.asarray(poles, dtype=np.complex128)
    zeros_array = np.asarray(zeros, dtype=np.complex128)
    dominant_pole_mag = float(np.min(-np.real(poles_array)))
    mean_pole_mag = float(np.mean(np.abs(poles_array)))

    if dominant_pole_mag <= 0.0:
        raise ValueError("Plant poles must be strictly stable.")

    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise ValueError("Plant coefficients must be finite.")

    return Plant(
        plant_id=plant_id,
        family=family,
        numerator=numerator,
        denominator=denominator,
        poles=poles_array,
        zeros=zeros_array,
        dc_gain=float(dc_gain),
        plant_order=len(poles),
        dominant_pole_mag=dominant_pole_mag,
        mean_pole_mag=mean_pole_mag,
    )


def _sample_first_order(rng: np.random.Generator, plant_id: int) -> Plant:
    pole = -rng.uniform(0.5, 3.5)
    dc_gain = rng.uniform(0.5, 2.0)
    return _make_transfer_function(plant_id, "first_order", [pole], [], dc_gain)


def _sample_second_order(
    rng: np.random.Generator,
    plant_id: int,
    family: str,
    damping_range: tuple[float, float],
) -> Plant:
    wn = rng.uniform(0.8, 4.0)
    zeta = rng.uniform(*damping_range)
    real = -zeta * wn
    imag = wn * np.sqrt(max(0.0, 1.0 - zeta**2))
    if imag > 1e-8:
        poles = [real + 1j * imag, real - 1j * imag]
    else:
        root = -wn * max(zeta, 0.4)
        poles = [root, root * rng.uniform(0.85, 1.15)]
    zeros: list[complex] = []
    if rng.random() < 0.25:
        zeros = [-rng.uniform(0.4, 4.0)]
    dc_gain = rng.uniform(0.5, 2.0)
    return _make_transfer_function(plant_id, family, poles, zeros, dc_gain)


def _sample_third_order(rng: np.random.Generator, plant_id: int) -> Plant:
    poles = sorted((-rng.uniform(0.4, 4.5, size=3)).tolist())
    zeros: list[complex] = []
    if rng.random() < 0.35:
        zeros = [-rng.uniform(0.4, 3.5)]
    dc_gain = rng.uniform(0.4, 1.8)
    return _make_transfer_function(plant_id, "third_order", poles, zeros, dc_gain)


def sample_plant(
    rng: np.random.Generator,
    plant_id: int,
    families: list[str] | None = None,
) -> Plant:
    available = families or FAMILY_CHOICES
    family = str(rng.choice(available))
    if family == "first_order":
        return _sample_first_order(rng, plant_id)
    if family == "second_order":
        return _sample_second_order(rng, plant_id, family, (0.35, 1.15))
    if family == "lightly_damped_second_order":
        return _sample_second_order(rng, plant_id, family, (0.08, 0.3))
    if family == "third_order":
        return _sample_third_order(rng, plant_id)
    raise ValueError(f"Unknown plant family: {family}")


def heuristic_pid_scales(plant: Plant) -> tuple[float, float, float]:
    process_gain = max(abs(plant.dc_gain), 0.1)
    dominant_time_constant = 1.0 / max(plant.dominant_pole_mag, 1e-3)
    kp = 1.0 / process_gain
    ki = kp / max(dominant_time_constant, 0.1)
    kd = 0.15 * kp * dominant_time_constant
    return kp, ki, kd


def log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def sample_pid_gains(
    rng: np.random.Generator,
    plant: Plant,
    kp_multiplier_range: tuple[float, float],
    ki_multiplier_range: tuple[float, float],
    kd_multiplier_range: tuple[float, float],
) -> tuple[float, float, float]:
    base_kp, base_ki, base_kd = heuristic_pid_scales(plant)
    kp = base_kp * log_uniform(rng, *kp_multiplier_range)
    ki = base_ki * log_uniform(rng, *ki_multiplier_range)
    kd = base_kd * log_uniform(rng, *kd_multiplier_range)
    return float(kp), float(ki), float(kd)


def plant_from_sample_row(row: dict[str, float] | np.ndarray | object) -> Plant:
    numerator = np.array(
        [float(row[f"num_{index}"]) for index in range(MAX_NUM_ORDER + 1)],
        dtype=np.float64,
    )
    denominator = np.array(
        [float(row[f"den_{index}"]) for index in range(MAX_DEN_ORDER + 1)],
        dtype=np.float64,
    )
    numerator = np.trim_zeros(numerator, trim="f")
    denominator = np.trim_zeros(denominator, trim="f")
    poles = np.roots(denominator)
    zeros = np.roots(numerator) if numerator.shape[0] > 1 else np.array([], dtype=np.complex128)
    return Plant(
        plant_id=int(row["plant_id"]),
        family=str(row["plant_family"]),
        numerator=numerator,
        denominator=denominator,
        poles=poles.astype(np.complex128),
        zeros=zeros.astype(np.complex128),
        dc_gain=float(row["dc_gain"]),
        plant_order=int(row["plant_order"]),
        dominant_pole_mag=float(row["dominant_pole_mag"]),
        mean_pole_mag=float(row["mean_pole_mag"]),
    )
