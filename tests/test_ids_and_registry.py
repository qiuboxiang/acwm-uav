from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from researchctl.ids import checkpoint_id, dataset_id, evaluation_id, run_id
from researchctl.schema import DatasetRecord
from researchctl.storage import Registry, RegistryError


def test_stable_ids() -> None:
    moment = datetime(2026, 8, 11, 1, 2, 3, tzinfo=UTC)
    assert dataset_id("Synthetic Regression", "1.0.0") == ("dataset:synthetic-regression@1.0.0")
    run = run_id("Main Ablation", "abcdef012345", moment)
    assert run == "run:20260811T010203Z-main-ablation-abcdef0"
    assert checkpoint_id(run, "best", "1" * 64).endswith("/best-11111111")
    assert evaluation_id(run, "robustness", moment).endswith("/robustness-20260811T010203Z")


def test_finalized_registry_record_is_immutable(tmp_path: Path) -> None:
    registry = Registry(tmp_path / "registry")
    created = datetime.now(UTC)
    record = DatasetRecord(
        id="dataset:demo@1",
        status="finalized",
        created_at=created,
        finalized_at=created,
        name="demo",
        version="1",
        source="generated://demo",
        license="CC0-1.0",
        splits={"train": "sha256:one"},
        preprocessing="none",
    )
    registry.write("datasets", record)
    registry.write("datasets", record)
    changed = record.model_copy(update={"preprocessing": "changed"})
    with pytest.raises(RegistryError, match="immutable"):
        registry.write("datasets", changed)
