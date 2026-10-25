"""Immutable registry persistence and cross-reference validation."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import TypeVar, cast

from researchctl.ids import filename_for_id
from researchctl.io import atomic_write_yaml, load_yaml
from researchctl.schema import (
    RECORD_MODELS,
    CheckpointRecord,
    DatasetRecord,
    EvaluationRecord,
    Record,
    RunRecord,
)

TRecord = TypeVar("TRecord", bound=Record)


class RegistryError(ValueError):
    pass


class Registry:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, kind: str, record_id: str) -> Path:
        if kind not in RECORD_MODELS:
            raise RegistryError(f"unknown registry kind: {kind}")
        return self.root / kind / filename_for_id(record_id)

    def write(self, kind: str, record: Record) -> Path:
        model = RECORD_MODELS.get(kind)
        if model is None or not isinstance(record, model):
            raise RegistryError(f"record does not match registry kind {kind}")
        path = self.path_for(kind, record.id)
        if path.exists():
            existing = model.model_validate(load_yaml(path))
            if existing.model_dump(mode="json") == record.model_dump(mode="json"):
                return path
            if existing.status == "finalized":
                raise RegistryError(f"finalized record is immutable: {record.id}")
        atomic_write_yaml(path, record)
        return path

    def load(self, kind: str, record_id: str, expected: type[TRecord] | None = None) -> TRecord:
        model = RECORD_MODELS.get(kind)
        if model is None:
            raise RegistryError(f"unknown registry kind: {kind}")
        path = self.path_for(kind, record_id)
        if not path.is_file():
            raise RegistryError(f"missing {kind} record: {record_id}")
        record = model.model_validate(load_yaml(path))
        if expected is not None and not isinstance(record, expected):
            raise RegistryError(f"record {record_id} has unexpected type")
        return cast(TRecord, record)

    def list(self, kind: str) -> builtins.list[Record]:
        model = RECORD_MODELS.get(kind)
        if model is None:
            raise RegistryError(f"unknown registry kind: {kind}")
        directory = self.root / kind
        if not directory.exists():
            return []
        records = []
        for path in sorted(directory.glob("*.yaml")):
            records.append(model.model_validate(load_yaml(path)))
        return records

    def datasets(self) -> dict[str, DatasetRecord]:
        return {item.id: item for item in self.list("datasets") if isinstance(item, DatasetRecord)}

    def runs(self) -> dict[str, RunRecord]:
        return {item.id: item for item in self.list("runs") if isinstance(item, RunRecord)}

    def checkpoints(self) -> dict[str, CheckpointRecord]:
        return {
            item.id: item for item in self.list("checkpoints") if isinstance(item, CheckpointRecord)
        }

    def evaluations(self) -> dict[str, EvaluationRecord]:
        return {
            item.id: item for item in self.list("evaluations") if isinstance(item, EvaluationRecord)
        }

    def validate_references(self) -> builtins.list[str]:
        errors: builtins.list[str] = []
        datasets = self.datasets()
        runs = self.runs()
        checkpoints = self.checkpoints()
        for run in runs.values():
            for reference in run.dataset_ids:
                if reference not in datasets:
                    errors.append(f"{run.id}: missing dataset {reference}")
            for reference in run.checkpoint_ids:
                if reference not in checkpoints:
                    errors.append(f"{run.id}: missing checkpoint {reference}")
        for checkpoint in checkpoints.values():
            if checkpoint.run_id not in runs:
                errors.append(f"{checkpoint.id}: missing run {checkpoint.run_id}")
        for evaluation in self.evaluations().values():
            if evaluation.run_id not in runs:
                errors.append(f"{evaluation.id}: missing run {evaluation.run_id}")
            if evaluation.checkpoint_id not in checkpoints:
                errors.append(f"{evaluation.id}: missing checkpoint {evaluation.checkpoint_id}")
            if evaluation.dataset_id not in datasets:
                errors.append(f"{evaluation.id}: missing dataset {evaluation.dataset_id}")
        return errors
