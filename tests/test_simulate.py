from __future__ import annotations

import numpy as np

from controlsimulator.plants import Plant
from controlsimulator.simulate import simulate_closed_loop


def test_closed_loop_simulation_known_case() -> None:
    plant = Plant(
        plant_id=0,
        family="known_first_order",
        numerator=np.array([1.0], dtype=np.float64),
        denominator=np.array([1.0, 1.0], dtype=np.float64),
        poles=np.array([-1.0 + 0.0j]),
        zeros=np.array([], dtype=np.complex128),
        dc_gain=1.0,
        plant_order=1,
        dominant_pole_mag=1.0,
        mean_pole_mag=1.0,
        min_damping_ratio=1.0,
        max_oscillation_hz=0.0,
        pole_spread_log10=0.0,
        has_complex_poles=False,
        max_real_part=-1.0,
        min_real_part=-1.0,
    )
    time_grid = np.linspace(0.0, 8.0, 200, dtype=np.float32)
    result = simulate_closed_loop(
        plant=plant,
        kp=1.5,
        ki=0.8,
        kd=0.1,
        tau_d=0.05,
        time_grid=time_grid,
    )

    assert result.stable is True
    assert result.trajectory is not None
    assert result.peak_control_effort is not None
    assert abs(float(result.trajectory[-1]) - 1.0) < 0.05
