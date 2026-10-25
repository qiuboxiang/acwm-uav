"""Small model used by the executable template test."""

from __future__ import annotations

from torch import Tensor, nn


class LinearRegressor(nn.Module):  # type: ignore[misc]
    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.linear(features)
