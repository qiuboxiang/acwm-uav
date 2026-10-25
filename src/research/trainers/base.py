"""Structured training results independent of orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingResult:
    metrics: dict[str, float]
    checkpoints: dict[str, Path]
