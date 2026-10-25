from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from researchctl.config import load_project
from researchctl.io import load_yaml, sha256_file
from researchctl.operations import register_dataset
from researchctl.paper import link_evidence
from researchctl.release import check_release
from researchctl.schema import CheckpointRecord, EvaluationRecord, ReleaseManifest, RunRecord
from researchctl.storage import Registry
from researchctl.validation import validate_repository


def _seed_trace(project: Path) -> tuple[str, str, str, str]:
    dataset = register_dataset(
        project,
        name="synthetic-regression",
        version="1.0.0",
        source="generated://synthetic",
        license_name="CC0-1.0",
        splits={"train": "sha256:train", "test": "sha256:test"},
        preprocessing="deterministic",
        redistributable=True,
        hf_revision="org/project-data@dev",
    )
    registry = Registry(project / load_project(project).registry_dir)
    now = datetime.now(UTC)
    run_id = "run:20260811T000000Z-demo-deadbee"
    checkpoint_path = project / "artifacts/checkpoints/demo/best.bin"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(b"deterministic checkpoint")
    checksum = sha256_file(checkpoint_path)
    checkpoint_id = f"ckpt:{run_id.removeprefix('run:')}/best-{checksum[:8]}"
    run = RunRecord(
        id=run_id,
        status="finalized",
        created_at=now,
        finalized_at=now,
        name="demo",
        command=["researchctl", "run", "train"],
        config_source="configs/experiment/synthetic.yaml",
        resolved_config={"seed": 7},
        code_commit="deadbeef",
        dirty=False,
        python_version="3.11.0",
        platform="test",
        seed=7,
        launcher={"kind": "local"},
        dataset_ids=[dataset.id],
        metrics={"val_mse": 0.1},
        checkpoint_ids=[checkpoint_id],
        output_dir="artifacts/checkpoints/demo",
    )
    checkpoint = CheckpointRecord(
        id=checkpoint_id,
        status="finalized",
        created_at=now,
        finalized_at=now,
        run_id=run.id,
        alias="best",
        format="test",
        path=checkpoint_path.relative_to(project).as_posix(),
        hf_revision="org/project-model@dev",
        sha256=checksum,
        size_bytes=checkpoint_path.stat().st_size,
        selection_metric="val_mse",
        selection_value=0.1,
    )
    evaluation_id = f"eval:{run_id.removeprefix('run:')}/default-20260811T000100Z"
    evaluation = EvaluationRecord(
        id=evaluation_id,
        status="finalized",
        created_at=now,
        finalized_at=now,
        run_id=run.id,
        checkpoint_id=checkpoint.id,
        dataset_id=dataset.id,
        suite="default",
        config={"metrics": ["mse"]},
        metrics={"mse": 0.1},
    )
    registry.write("runs", run)
    registry.write("checkpoints", checkpoint)
    registry.write("evaluations", evaluation)
    table = project / "artifacts/evals/main-table.csv"
    table.write_text("metric,value\nmse,0.1\n", encoding="utf-8")
    link_evidence(
        project,
        label="tab:main",
        kind="table",
        source=table,
        generator="tests/test_traceability.py",
        run_ids=[run.id],
        evaluation_ids=[evaluation.id],
        dataset_ids=[dataset.id],
        checkpoint_ids=[checkpoint.id],
    )
    return dataset.id, run.id, checkpoint.id, evaluation.id


def test_complete_trace_refreshes_release(project: Path) -> None:
    dataset, run, checkpoint, evaluation = _seed_trace(project)
    manifest = ReleaseManifest.model_validate(load_yaml(project / "releases/current.yaml"))
    assert manifest.datasets == [dataset]
    assert manifest.runs == [run]
    assert manifest.checkpoints == [checkpoint]
    assert manifest.evaluations == [evaluation]
    assert manifest.evidence == ["tab:main"]
    assert validate_repository(project) == []
    assert check_release(project) == []


def test_strict_release_reports_operational_gates(project: Path) -> None:
    _seed_trace(project)
    errors = check_release(project, strict=True)
    assert any("Git commit" in error for error in errors)
    assert any("Overleaf HEAD" in error for error in errors)


def test_unlinked_inventory_does_not_enter_release(project: Path) -> None:
    dataset = register_dataset(
        project,
        name="unused",
        version="1",
        source="generated://unused",
        license_name="CC0-1.0",
        splits={"train": "sha256:unused"},
        preprocessing="none",
    )
    manifest = ReleaseManifest.model_validate(load_yaml(project / "releases/current.yaml"))
    assert dataset.id not in manifest.datasets
    assert dataset.id in manifest.unlinked["datasets"]
