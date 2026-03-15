from __future__ import annotations

import numpy as np

from controlsimulator.metrics import extract_response_metrics


def test_metric_extraction_on_first_order_curve() -> None:
    time_grid = np.linspace(0.0, 8.0, 400, dtype=np.float64)
    trajectory = 1.0 - np.exp(-time_grid)
    metrics = extract_response_metrics(time_grid, trajectory)

    assert metrics.overshoot_pct == 0.0
    assert abs(metrics.rise_time - 2.2) < 0.1
    assert abs(metrics.settling_time - 3.9) < 0.1
    assert metrics.steady_state_error < 0.01
