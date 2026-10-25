from __future__ import annotations

from pathlib import Path

import pytest

from researchctl.operations import register_dataset, train_experiment


@pytest.mark.research
def test_optional_pytorch_training_loop(project: Path) -> None:
    pytest.importorskip("torch")
    register_dataset(
        project,
        name="synthetic-regression",
        version="1.0.0",
        source="generated://synthetic",
        license_name="CC0-1.0",
        splits={"train": "sha256:train", "test": "sha256:test"},
        preprocessing="deterministic",
    )
    record = train_experiment(project, project / "configs/experiment/synthetic.yaml")
    assert record is not None
    assert record.status == "finalized"
    assert len(record.checkpoint_ids) == 2
