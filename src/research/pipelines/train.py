"""Compose model, data, and trainer from a resolved configuration."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from research.pipelines.common import instantiate
from research.trainers import TorchTrainer, TrainingResult


def run_training(config: dict[str, Any], output_dir: Path) -> TrainingResult:
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    model = instantiate(dict(config["model"]))
    data = instantiate(dict(config["data"]))
    trainer = TorchTrainer(**dict(config["trainer"]))
    return trainer.fit(model, data.train_loader(), data.test_loader(), output_dir)
