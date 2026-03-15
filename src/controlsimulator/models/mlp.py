from __future__ import annotations

import torch
from torch import nn


class FeedForwardBackbone(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden_dim in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = previous

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TrajectoryRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = FeedForwardBackbone(input_dim, hidden_sizes, dropout)
        self.head = nn.Linear(self.backbone.output_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(inputs)
        return self.head(hidden)
