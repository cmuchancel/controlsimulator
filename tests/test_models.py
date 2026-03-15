from __future__ import annotations

import torch

from controlsimulator.features import FEATURE_COLUMNS
from controlsimulator.models import StabilityClassifier, TrajectoryRegressor


def test_model_forward_shapes() -> None:
    input_dim = len(FEATURE_COLUMNS)
    inputs = torch.randn(5, input_dim)
    classifier = StabilityClassifier(input_dim=input_dim, hidden_sizes=[32, 16], dropout=0.1)
    regressor = TrajectoryRegressor(
        input_dim=input_dim,
        output_dim=80,
        hidden_sizes=[64, 64],
        dropout=0.1,
    )

    assert classifier(inputs).shape == (5,)
    assert regressor(inputs).shape == (5, 80)
