from __future__ import annotations

import torch
from torch import nn

from controlsimulator.models.mlp import FeedForwardBackbone


class StabilityClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float) -> None:
        super().__init__()
        self.backbone = FeedForwardBackbone(input_dim, hidden_sizes, dropout)
        self.head = nn.Linear(self.backbone.output_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(inputs)
        return self.head(hidden).squeeze(-1)
