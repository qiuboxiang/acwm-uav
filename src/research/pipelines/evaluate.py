"""Evaluate a registered checkpoint against configured data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from research.evaluation.base import EvaluationResult
from research.pipelines.common import instantiate


def evaluate_checkpoint(
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    checkpoint_path: Path,
    device: str = "cpu",
) -> EvaluationResult:
    model = instantiate(model_config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    data = instantiate(data_config)
    squared_error = 0.0
    sample_count = 0
    with torch.no_grad():
        for features, targets in data.test_loader():
            prediction = model(features.to(device)).cpu()
            squared_error += float(((prediction - targets) ** 2).sum().item())
            sample_count += targets.numel()
    return EvaluationResult(metrics={"mse": squared_error / max(sample_count, 1)})
