"""Reference metrics."""

from __future__ import annotations

from torch import Tensor


def mean_squared_error(prediction: Tensor, target: Tensor) -> float:
    return float(((prediction - target) ** 2).mean().item())
