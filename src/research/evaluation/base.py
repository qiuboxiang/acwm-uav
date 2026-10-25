"""Evaluation result contract."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
