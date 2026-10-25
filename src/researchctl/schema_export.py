"""Deterministic JSON Schema artifacts for external tooling and CI."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from researchctl.io import atomic_write_text
from researchctl.schema import (
    CheckpointRecord,
    DatasetRecord,
    EvaluationRecord,
    EvidenceDocument,
    OverleafConfig,
    ProjectSettings,
    ReleaseManifest,
    RunRecord,
    VenueRecord,
)

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "project": ProjectSettings,
    "dataset": DatasetRecord,
    "run": RunRecord,
    "checkpoint": CheckpointRecord,
    "evaluation": EvaluationRecord,
    "evidence": EvidenceDocument,
    "venue": VenueRecord,
    "overleaf": OverleafConfig,
    "release": ReleaseManifest,
}


def render_schemas() -> dict[str, str]:
    return {
        f"{name}.schema.json": json.dumps(
            model.model_json_schema(), indent=2, sort_keys=True, ensure_ascii=False
        )
        + "\n"
        for name, model in SCHEMA_MODELS.items()
    }


def export_schemas(root: Path) -> list[Path]:
    directory = root / "schemas"
    paths: list[Path] = []
    for name, content in render_schemas().items():
        path = directory / name
        atomic_write_text(path, content)
        paths.append(path)
    return paths
