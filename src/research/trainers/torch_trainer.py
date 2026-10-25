"""Transparent PyTorch training loop with rank-safe checkpointing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from research.trainers.base import TrainingResult


def _distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device)
    if _distributed():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        tensor /= torch.distributed.get_world_size()
    return float(tensor.item())


def _distributed_loader(loader: DataLoader, shuffle: bool) -> DataLoader:
    if not _distributed():
        return loader
    sampler = DistributedSampler(loader.dataset, shuffle=shuffle)
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        drop_last=loader.drop_last,
    )


class TorchTrainer:
    def __init__(
        self,
        epochs: int,
        learning_rate: float,
        device: str = "auto",
        precision: str = "32",
        selection_metric: str = "val_mse",
        selection_mode: str = "min",
        log_every: int = 10,
        **_: Any,
    ) -> None:
        if precision != "32":
            raise ValueError("the reference trainer currently supports precision='32' only")
        if selection_mode not in {"min", "max"}:
            raise ValueError("selection_mode must be 'min' or 'max'")
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device_name = device
        self.selection_metric = selection_metric
        self.selection_mode = selection_mode
        self.log_every = log_every

    def _device(self) -> torch.device:
        if self.device_name != "auto":
            return torch.device(self.device_name)
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            return torch.device("cuda", local_rank)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        output_dir: Path,
    ) -> TrainingResult:
        device = self._device()
        if _distributed() and not torch.distributed.is_initialized():
            backend = "nccl" if device.type == "cuda" else "gloo"
            torch.distributed.init_process_group(backend=backend)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model.to(device)
        wrapped: nn.Module = model
        if _distributed():
            wrapped = DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None
            )
        train_loader = _distributed_loader(train_loader, shuffle=True)
        validation_loader = _distributed_loader(validation_loader, shuffle=False)
        optimizer = torch.optim.SGD(wrapped.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        output_dir.mkdir(parents=True, exist_ok=True)
        best_value = float("inf") if self.selection_mode == "min" else float("-inf")
        best_path = output_dir / "best.pt"
        last_path = output_dir / "last.pt"
        final_train = 0.0
        final_validation = 0.0

        for epoch in range(self.epochs):
            sampler = getattr(train_loader, "sampler", None)
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            wrapped.train()
            train_total = 0.0
            batches = 0
            for features, targets in train_loader:
                features, targets = features.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(wrapped(features), targets)
                loss.backward()
                optimizer.step()
                train_total += float(loss.item())
                batches += 1
            final_train = _reduce_mean(train_total / max(batches, 1), device)

            wrapped.eval()
            validation_total = 0.0
            validation_batches = 0
            with torch.no_grad():
                for features, targets in validation_loader:
                    features, targets = features.to(device), targets.to(device)
                    validation_total += float(loss_fn(wrapped(features), targets).item())
                    validation_batches += 1
            final_validation = _reduce_mean(validation_total / max(validation_batches, 1), device)
            improved = (
                final_validation < best_value
                if self.selection_mode == "min"
                else final_validation > best_value
            )
            if improved:
                best_value = final_validation
                if _rank() == 0:
                    torch.save(model.state_dict(), best_path)

        if _rank() == 0:
            torch.save(model.state_dict(), last_path)
        if _distributed():
            torch.distributed.barrier()
        return TrainingResult(
            metrics={"train_mse": final_train, "val_mse": final_validation},
            checkpoints={"best": best_path, "last": last_path},
        )
