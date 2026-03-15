from __future__ import annotations

import torch

from controlsimulator.models import StabilityClassifier, TrajectoryRegressor


def test_model_forward_shapes() -> None:
    inputs = torch.randn(5, 14)
    classifier = StabilityClassifier(input_dim=14, hidden_sizes=[32, 16], dropout=0.1)
    regressor = TrajectoryRegressor(
        input_dim=14,
        output_dim=80,
        hidden_sizes=[64, 64],
        dropout=0.1,
    )

    assert classifier(inputs).shape == (5,)
    assert regressor(inputs).shape == (5, 80)
