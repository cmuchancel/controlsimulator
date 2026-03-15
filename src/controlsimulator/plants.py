from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_DEN_ORDER = 4
MAX_NUM_ORDER = 2
MAX_OSCILLATORY_WN = 38.0
FAMILY_CHOICES = [
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
    min_damping_ratio: float
    max_oscillation_hz: float
    pole_spread_log10: float
    has_complex_poles: bool
    max_real_part: float
    min_real_part: float

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


def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _sample_real_pole(rng: np.random.Generator, low: float, high: float) -> float:
    return -_log_uniform(rng, low, high)


def _sample_dc_gain(rng: np.random.Generator, low: float = 0.35, high: float = 2.2) -> float:
    return float(rng.uniform(low, high))


def _sample_real_zero(
    rng: np.random.Generator,
    low: float = 0.03,
    high: float = 20.0,
) -> complex:
    return -_log_uniform(rng, low, high)


def _complex_pair(wn: float, zeta: float) -> list[complex]:
    wn = float(min(max(wn, 0.02), MAX_OSCILLATORY_WN))
    zeta = float(np.clip(zeta, 0.01, 1.8))
    real = -zeta * wn
    imag = wn * np.sqrt(max(0.0, 1.0 - min(zeta, 0.999999) ** 2))
    return [real + 1j * imag, real - 1j * imag]


def _complex_pair_from_parts(real: float, imag: float) -> list[complex]:
    return [complex(real, imag), complex(real, -imag)]


def _damping_ratio(pole: complex) -> float:
    if abs(np.imag(pole)) < 1e-9:
        return 1.0
    return float(np.clip(-np.real(pole) / max(abs(pole), 1e-9), 0.0, 1.0))


def _sample_real_poles(
    rng: np.random.Generator,
    count: int,
    low: float,
    high: float,
) -> list[complex]:
    magnitudes = np.sort(np.exp(rng.uniform(np.log(low), np.log(high), size=count)))
    return [-float(value) for value in magnitudes]


def _single_zero_list(
    rng: np.random.Generator,
    probability: float,
    low: float = 0.03,
    high: float = 20.0,
) -> list[complex]:
    if rng.random() >= probability:
        return []
    return [_sample_real_zero(rng, low, high)]


def _make_transfer_function(
    plant_id: int,
    family: str,
    poles: list[complex],
    zeros: list[complex],
    dc_gain: float,
    allow_unstable_poles: bool = False,
) -> Plant:
    denominator = np.poly(poles).real.astype(np.float64)
    base_numerator = np.poly(zeros).real.astype(np.float64) if zeros else np.array([1.0])
    leading = float(denominator[0])
    denominator = denominator / leading
    numerator = (dc_gain * denominator[-1] / max(base_numerator[-1], 1e-12)) * base_numerator
    numerator = numerator / leading

    if numerator.shape[0] > MAX_NUM_ORDER + 1 or denominator.shape[0] > MAX_DEN_ORDER + 1:
        raise ValueError("Sampled plant exceeded configured polynomial padding limits.")
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise ValueError("Plant coefficients must be finite.")

    coeffs = np.concatenate([np.abs(numerator), np.abs(denominator)])
    nonzero = coeffs[coeffs > 1e-10]
    if nonzero.size and (nonzero.max() / nonzero.min()) > 1e8:
        raise ValueError("Ill-conditioned transfer-function coefficients.")

    poles_array = np.asarray(poles, dtype=np.complex128)
    zeros_array = np.asarray(zeros, dtype=np.complex128)
    real_parts = np.real(poles_array)
    if not allow_unstable_poles and np.any(real_parts >= 0.0):
        raise ValueError("Plant poles must be strictly stable.")

    pole_magnitudes = np.abs(poles_array)
    dominant_pole_mag = float(np.min(np.abs(real_parts)))
    mean_pole_mag = float(np.mean(pole_magnitudes))
    min_damping_ratio = float(np.min([_damping_ratio(pole) for pole in poles_array]))
    max_oscillation_hz = float(np.max(np.abs(np.imag(poles_array))) / (2.0 * np.pi))
    pole_spread_log10 = float(
        np.log10(np.max(pole_magnitudes) / max(np.min(pole_magnitudes), 1e-8))
    )
    has_complex_poles = bool(np.any(np.abs(np.imag(poles_array)) > 1e-8))

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
        min_damping_ratio=min_damping_ratio,
        max_oscillation_hz=max_oscillation_hz,
        pole_spread_log10=pole_spread_log10,
        has_complex_poles=has_complex_poles,
        max_real_part=float(np.max(real_parts)),
        min_real_part=float(np.min(real_parts)),
    )


def _make_constant_numerator_plant(
    plant_id: int,
    family: str,
    poles: list[complex],
    b0: float,
    allow_unstable_poles: bool = False,
) -> Plant:
    denominator = np.poly(poles).real.astype(np.float64)
    if denominator.shape[0] != 4:
        raise ValueError("Third-order campaign plants must have exactly three poles.")
    a0 = float(denominator[-1])
    if abs(a0) < 1e-9:
        raise ValueError("Degenerate third-order denominator.")
    return _make_transfer_function(
        plant_id=plant_id,
        family=family,
        poles=poles,
        zeros=[],
        dc_gain=float(b0 / a0),
        allow_unstable_poles=allow_unstable_poles,
    )


def _sample_first_order(rng: np.random.Generator, plant_id: int) -> Plant:
    pole = _sample_real_pole(rng, 0.05, 12.0)
    zeros = _single_zero_list(rng, probability=0.18, low=0.04, high=8.0)
    return _make_transfer_function(
        plant_id,
        "first_order",
        [pole],
        zeros,
        _sample_dc_gain(rng),
    )


def _sample_second_order(
    rng: np.random.Generator,
    plant_id: int,
    family: str,
    damping_range: tuple[float, float],
    wn_range: tuple[float, float],
    zero_probability: float = 0.28,
) -> Plant:
    zeta = float(rng.uniform(*damping_range))
    wn = _log_uniform(rng, *wn_range)
    zeros = _single_zero_list(rng, probability=zero_probability, low=0.04, high=18.0)
    return _make_transfer_function(
        plant_id,
        family,
        _complex_pair(wn, zeta),
        zeros,
        _sample_dc_gain(rng),
    )


def _sample_overdamped_second_order(rng: np.random.Generator, plant_id: int) -> Plant:
    slow = _log_uniform(rng, 0.03, 3.0)
    ratio = float(rng.uniform(1.8, 20.0))
    poles = [-slow, -(slow * ratio)]
    zeros = _single_zero_list(rng, probability=0.22, low=0.03, high=6.0)
    return _make_transfer_function(
        plant_id,
        "overdamped_second_order",
        poles,
        zeros,
        _sample_dc_gain(rng),
    )


def _sample_third_order_real_poles(rng: np.random.Generator, plant_id: int) -> Plant:
    poles = _sample_real_poles(rng, count=3, low=0.03, high=45.0)
    zeros = _single_zero_list(rng, probability=0.35, low=0.05, high=12.0)
    return _make_transfer_function(
        plant_id,
        "third_order_real_poles",
        poles,
        zeros,
        _sample_dc_gain(rng, 0.3, 1.9),
    )


def _sample_third_order_mixed_real_complex(rng: np.random.Generator, plant_id: int) -> Plant:
    wn = _log_uniform(rng, 0.12, 18.0)
    zeta = float(rng.uniform(0.18, 0.78))
    real_pole = _sample_real_pole(rng, 0.05, 50.0)
    zeros = _single_zero_list(rng, probability=0.35, low=0.05, high=10.0)
    return _make_transfer_function(
        plant_id,
        "third_order_mixed_real_complex",
        [*_complex_pair(wn, zeta), real_pole],
        zeros,
        _sample_dc_gain(rng, 0.3, 1.9),
    )


def _sample_weakly_resonant_third_order(rng: np.random.Generator, plant_id: int) -> Plant:
    wn = _log_uniform(rng, 0.4, 24.0)
    zeta = float(rng.uniform(0.05, 0.18))
    real_pole = _sample_real_pole(rng, 0.8, 60.0)
    zeros = _single_zero_list(rng, probability=0.42, low=0.05, high=8.0)
    return _make_transfer_function(
        plant_id,
        "weakly_resonant_third_order",
        [*_complex_pair(wn, zeta), real_pole],
        zeros,
        _sample_dc_gain(rng, 0.3, 1.8),
    )


def _sample_highly_resonant_second_order(rng: np.random.Generator, plant_id: int) -> Plant:
    return _sample_second_order(
        rng,
        plant_id,
        "highly_resonant_second_order",
        damping_range=(0.02, 0.08),
        wn_range=(0.8, 35.0),
        zero_probability=0.25,
    )


def _sample_fourth_order_real(rng: np.random.Generator, plant_id: int) -> Plant:
    poles = _sample_real_poles(rng, count=4, low=0.02, high=70.0)
    zeros: list[complex] = []
    if rng.random() < 0.45:
        zeros.append(_sample_real_zero(rng, 0.05, 15.0))
    if rng.random() < 0.22:
        zeros.append(_sample_real_zero(rng, 0.08, 18.0))
    return _make_transfer_function(
        plant_id,
        "fourth_order_real",
        poles,
        zeros,
        _sample_dc_gain(rng, 0.25, 1.8),
    )


def _sample_fourth_order_mixed_complex(rng: np.random.Generator, plant_id: int) -> Plant:
    wn = _log_uniform(rng, 0.15, 22.0)
    zeta = float(rng.uniform(0.12, 0.82))
    real_poles = _sample_real_poles(rng, count=2, low=0.04, high=60.0)
    zeros: list[complex] = []
    if rng.random() < 0.4:
        zeros.append(_sample_real_zero(rng, 0.05, 12.0))
    if rng.random() < 0.16:
        zeros.append(_sample_real_zero(rng, 0.08, 18.0))
    return _make_transfer_function(
        plant_id,
        "fourth_order_mixed_complex",
        [*_complex_pair(wn, zeta), *real_poles],
        zeros,
        _sample_dc_gain(rng, 0.25, 1.8),
    )


def _sample_two_mode_resonant(rng: np.random.Generator, plant_id: int) -> Plant:
    low_wn = _log_uniform(rng, 0.3, 5.5)
    ratio = float(rng.uniform(1.8, 4.8))
    high_wn = min(low_wn * ratio, MAX_OSCILLATORY_WN)
    low_zeta = float(rng.uniform(0.04, 0.18))
    high_zeta = float(rng.uniform(0.08, 0.32))
    zeros = _single_zero_list(rng, probability=0.25, low=0.05, high=8.0)
    return _make_transfer_function(
        plant_id,
        "two_mode_resonant",
        [*_complex_pair(low_wn, low_zeta), *_complex_pair(high_wn, high_zeta)],
        zeros,
        _sample_dc_gain(rng, 0.3, 1.6),
    )


def _sample_near_integrator(rng: np.random.Generator, plant_id: int) -> Plant:
    slow = _log_uniform(rng, 0.01, 0.08)
    poles: list[complex] = [-slow]
    if rng.random() < 0.8:
        poles.append(_sample_real_pole(rng, 0.25, 4.0))
    zeros = _single_zero_list(rng, probability=0.12, low=0.02, high=0.8)
    return _make_transfer_function(
        plant_id,
        "near_integrator",
        poles,
        zeros,
        _sample_dc_gain(rng, 0.4, 1.7),
    )


def _sample_slow_dynamics_family(rng: np.random.Generator, plant_id: int) -> Plant:
    if rng.random() < 0.55:
        return _sample_second_order(
            rng,
            plant_id,
            "slow_dynamics_family",
            damping_range=(0.18, 0.9),
            wn_range=(0.03, 0.28),
            zero_probability=0.18,
        )
    poles = _sample_real_poles(rng, count=3, low=0.01, high=0.35)
    zeros = _single_zero_list(rng, probability=0.18, low=0.02, high=0.6)
    return _make_transfer_function(
        plant_id,
        "slow_dynamics_family",
        poles,
        zeros,
        _sample_dc_gain(rng, 0.35, 1.8),
    )


def _sample_fast_dynamics_family(rng: np.random.Generator, plant_id: int) -> Plant:
    if rng.random() < 0.5:
        return _sample_second_order(
            rng,
            plant_id,
            "fast_dynamics_family",
            damping_range=(0.12, 0.65),
            wn_range=(8.0, 38.0),
            zero_probability=0.22,
        )
    if rng.random() < 0.65:
        return _make_transfer_function(
            plant_id,
            "fast_dynamics_family",
            _sample_real_poles(rng, count=3, low=10.0, high=100.0),
            _single_zero_list(rng, probability=0.2, low=6.0, high=30.0),
            _sample_dc_gain(rng, 0.3, 1.8),
        )
    return _make_transfer_function(
        plant_id,
        "fast_dynamics_family",
        [
            *_complex_pair(_log_uniform(rng, 10.0, 38.0), float(rng.uniform(0.08, 0.28))),
            _sample_real_pole(rng, 12.0, 100.0),
        ],
        _single_zero_list(rng, probability=0.2, low=6.0, high=28.0),
        _sample_dc_gain(rng, 0.3, 1.8),
    )


def _sample_campaign_third_order_stable(rng: np.random.Generator, plant_id: int) -> Plant:
    poles = [complex(float(rng.uniform(-5.0, -0.05)), 0.0) for _ in range(3)]
    b0 = float(rng.uniform(0.5, 3.0))
    return _make_constant_numerator_plant(
        plant_id,
        "campaign_third_order_stable",
        poles,
        b0,
    )


def _sample_campaign_third_order_oscillatory(
    rng: np.random.Generator,
    plant_id: int,
) -> Plant:
    real = float(rng.uniform(-0.5, -0.01))
    imag = float(rng.uniform(0.5, 5.0))
    real_pole = complex(float(rng.uniform(-5.0, -0.05)), 0.0)
    b0 = float(rng.uniform(0.5, 3.0))
    return _make_constant_numerator_plant(
        plant_id,
        "campaign_third_order_oscillatory",
        [*_complex_pair_from_parts(real, imag), real_pole],
        b0,
    )


def _sample_campaign_third_order_ood_lightly_damped(
    rng: np.random.Generator,
    plant_id: int,
) -> Plant:
    real = float(rng.uniform(-0.08, -0.005))
    imag = float(rng.uniform(1.0, 6.0))
    real_pole = complex(float(rng.uniform(-1.5, -0.03)), 0.0)
    b0 = float(rng.uniform(0.5, 3.0))
    return _make_constant_numerator_plant(
        plant_id,
        "campaign_third_order_ood_lightly_damped",
        [*_complex_pair_from_parts(real, imag), real_pole],
        b0,
    )


def _sample_campaign_third_order_near_instability(
    rng: np.random.Generator,
    plant_id: int,
) -> Plant:
    poles = [complex(float(rng.uniform(-0.05, -1e-4)), 0.0) for _ in range(3)]
    b0 = float(rng.uniform(0.5, 3.0))
    return _make_constant_numerator_plant(
        plant_id,
        "campaign_third_order_near_instability",
        poles,
        b0,
        allow_unstable_poles=True,
    )


def _sample_campaign_third_order_unstable(rng: np.random.Generator, plant_id: int) -> Plant:
    unstable_pole = complex(float(rng.uniform(1e-4, 0.5)), 0.0)
    stable_poles = [complex(float(rng.uniform(-5.0, -0.05)), 0.0) for _ in range(2)]
    b0 = float(rng.uniform(0.5, 3.0))
    return _make_constant_numerator_plant(
        plant_id,
        "campaign_third_order_unstable",
        [unstable_pole, *stable_poles],
        b0,
        allow_unstable_poles=True,
    )


FAMILY_SAMPLERS: dict[str, Callable[[np.random.Generator, int], Plant]] = {
    "first_order": _sample_first_order,
    "second_order": lambda rng, plant_id: _sample_second_order(
        rng,
        plant_id,
        "second_order",
        damping_range=(0.55, 0.95),
        wn_range=(0.08, 9.0),
    ),
    "underdamped_second_order": lambda rng, plant_id: _sample_second_order(
        rng,
        plant_id,
        "underdamped_second_order",
        damping_range=(0.18, 0.55),
        wn_range=(0.1, 14.0),
    ),
    "overdamped_second_order": _sample_overdamped_second_order,
    "lightly_damped_second_order": lambda rng, plant_id: _sample_second_order(
        rng,
        plant_id,
        "lightly_damped_second_order",
        damping_range=(0.08, 0.18),
        wn_range=(0.2, 20.0),
    ),
    "highly_resonant_second_order": _sample_highly_resonant_second_order,
    "third_order": lambda rng, plant_id: (
        _sample_third_order_real_poles(rng, plant_id)
        if rng.random() < 0.5
        else _sample_third_order_mixed_real_complex(rng, plant_id)
    ),
    "third_order_real_poles": _sample_third_order_real_poles,
    "third_order_mixed_real_complex": _sample_third_order_mixed_real_complex,
    "weakly_resonant_third_order": _sample_weakly_resonant_third_order,
    "fourth_order_real": _sample_fourth_order_real,
    "fourth_order_mixed_complex": _sample_fourth_order_mixed_complex,
    "two_mode_resonant": _sample_two_mode_resonant,
    "near_integrator": _sample_near_integrator,
    "slow_dynamics_family": _sample_slow_dynamics_family,
    "fast_dynamics_family": _sample_fast_dynamics_family,
    "campaign_third_order_stable": _sample_campaign_third_order_stable,
    "campaign_third_order_oscillatory": _sample_campaign_third_order_oscillatory,
    "campaign_third_order_ood_lightly_damped": _sample_campaign_third_order_ood_lightly_damped,
    "campaign_third_order_near_instability": _sample_campaign_third_order_near_instability,
    "campaign_third_order_unstable": _sample_campaign_third_order_unstable,
}


def sample_plant(
    rng: np.random.Generator,
    plant_id: int,
    families: list[str] | None = None,
    family_sampling_weights: dict[str, float] | None = None,
) -> Plant:
    available = families or FAMILY_CHOICES
    if family_sampling_weights:
        weights = np.asarray(
            [max(float(family_sampling_weights.get(family, 0.0)), 0.0) for family in available],
            dtype=np.float64,
        )
        if weights.sum() <= 0.0:
            raise ValueError("family_sampling_weights must assign positive mass.")
        family = str(rng.choice(available, p=weights / weights.sum()))
    else:
        family = str(rng.choice(available))
    sampler = FAMILY_SAMPLERS.get(family)
    if sampler is None:
        raise ValueError(f"Unknown plant family: {family}")

    last_error: Exception | None = None
    for _ in range(64):
        try:
            return sampler(rng, plant_id)
        except ValueError as error:
            last_error = error
    raise RuntimeError(f"Failed to sample a valid plant for family {family}: {last_error}")


def heuristic_pid_scales(plant: Plant) -> tuple[float, float, float]:
    process_gain = max(abs(plant.dc_gain), 0.1)
    dominant_time_constant = 1.0 / max(plant.dominant_pole_mag, 1e-3)
    dominant_time_constant = float(np.clip(dominant_time_constant, 0.01, 100.0))
    damping_factor = 0.8 + (0.35 * (1.0 - min(plant.min_damping_ratio, 1.0)))
    kp = damping_factor / process_gain
    ki = kp / max(dominant_time_constant, 0.02)
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


def _row_value(row: dict[str, Any] | np.ndarray | object, key: str, default: float = 0.0) -> float:
    if isinstance(row, dict):
        return float(row.get(key, default))
    if hasattr(row, "get"):
        try:
            value = row.get(key, default)
            if value is None:
                return float(default)
            return float(value)
        except TypeError:
            pass
    try:
        return float(row[key])  # type: ignore[index]
    except Exception:
        return float(default)


def _trim_polynomial(values: np.ndarray, default: float = 1.0) -> np.ndarray:
    trimmed = np.trim_zeros(values, trim="f")
    if trimmed.size == 0:
        return np.array([default], dtype=np.float64)
    return trimmed.astype(np.float64)


def plant_from_sample_row(row: dict[str, float] | np.ndarray | object) -> Plant:
    has_polynomial_columns = any(
        np.isfinite(_row_value(row, f"den_{index}", default=float("nan")))
        for index in range(MAX_DEN_ORDER + 1)
    )
    if has_polynomial_columns:
        numerator = np.array(
            [_row_value(row, f"num_{index}", 0.0) for index in range(MAX_NUM_ORDER + 1)],
            dtype=np.float64,
        )
        denominator = np.array(
            [_row_value(row, f"den_{index}", 0.0) for index in range(MAX_DEN_ORDER + 1)],
            dtype=np.float64,
        )
        numerator = _trim_polynomial(numerator, default=1.0)
        denominator = _trim_polynomial(denominator, default=1.0)
    else:
        numerator = np.array([_row_value(row, "b0", default=1.0)], dtype=np.float64)
        denominator = np.array(
            [
                1.0,
                _row_value(row, "a2", default=0.0),
                _row_value(row, "a1", default=0.0),
                _row_value(row, "a0", default=1.0),
            ],
            dtype=np.float64,
        )
    poles = np.roots(denominator)
    zeros = np.roots(numerator) if numerator.shape[0] > 1 else np.array([], dtype=np.complex128)
    return Plant(
        plant_id=int(_row_value(row, "plant_id")),
        family=str(getattr(row, "get", lambda *_: "unknown")("plant_family", "unknown")),
        numerator=numerator,
        denominator=denominator,
        poles=poles.astype(np.complex128),
        zeros=zeros.astype(np.complex128),
        dc_gain=_row_value(row, "dc_gain", default=1.0),
        plant_order=int(_row_value(row, "plant_order", default=max(1, denominator.shape[0] - 1))),
        dominant_pole_mag=_row_value(
            row,
            "dominant_pole_mag",
            default=float(np.min(-np.real(poles))) if poles.size else 1.0,
        ),
        mean_pole_mag=_row_value(
            row,
            "mean_pole_mag",
            default=float(np.mean(np.abs(poles))) if poles.size else 1.0,
        ),
        min_damping_ratio=_row_value(
            row,
            "plant_min_damping_ratio",
            default=float(np.min([_damping_ratio(pole) for pole in poles])) if poles.size else 1.0,
        ),
        max_oscillation_hz=_row_value(
            row,
            "plant_max_oscillation_hz",
            default=float(np.max(np.abs(np.imag(poles))) / (2.0 * np.pi)) if poles.size else 0.0,
        ),
        pole_spread_log10=_row_value(
            row,
            "plant_pole_spread_log10",
            default=float(
                np.log10(np.max(np.abs(poles)) / max(np.min(np.abs(poles)), 1e-8))
            )
            if poles.size
            else 0.0,
        ),
        has_complex_poles=bool(_row_value(row, "plant_has_complex_poles", default=0.0)),
        max_real_part=_row_value(
            row,
            "plant_max_real_part",
            default=float(np.max(np.real(poles))) if poles.size else -1.0,
        ),
        min_real_part=_row_value(
            row,
            "plant_min_real_part",
            default=float(np.min(np.real(poles))) if poles.size else -1.0,
        ),
    )
