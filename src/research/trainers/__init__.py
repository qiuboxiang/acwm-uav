"""Training loops."""

from research.trainers.base import TrainingResult
from research.trainers.torch_trainer import TorchTrainer

__all__ = ["TorchTrainer", "TrainingResult"]
