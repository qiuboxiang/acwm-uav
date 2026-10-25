"""Repository-wide invariant checks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from researchctl.config import load_project
from researchctl.docs import render_registry_docs
from researchctl.io import load_yaml, sha256_file
from researchctl.schema import EvidenceDocument, OverleafConfig, ReleaseManifest, VenueRecord
from researchctl.schema_export import render_schemas
from researchctl.storage import Registry

IGNORED_DIRECTORIES = {
    ".cache",
    ".dvc",
    ".git",
    ".mlruns",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".runs",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "sync-conflicts",
}
PRIVATE_PATTERNS = [
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    re.compile(r"(?i)(token|password|secret)\s*:\s*[^\s\"']{8,}"),
]


def stable_directories(root: Path) -> Iterable[Path]:
    for path in sorted(item for item in root.rglob("*") if item.is_dir()):
        relative = path.relative_to(root)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        if relative.parts[0] == "artifacts" and len(relative.parts) > 2:
            continue
        yield path


def validate_agents(root: Path) -> list[str]:
    errors: list[str] = []
    if not (root / "AGENTS.md").is_file():
        errors.append("root AGENTS.md is missing")
    for directory in stable_directories(root):
        if not (directory / "AGENTS.md").is_file():
            errors.append(f"missing AGENTS.md: {directory.relative_to(root)}")
        size = 0
        current = directory
        while True:
            instructions = current / "AGENTS.md"
            if instructions.is_file():
                size += instructions.stat().st_size
            if current == root:
                break
            current = current.parent
        if size > 65536:
            errors.append(f"instruction chain exceeds 65536 bytes: {directory.relative_to(root)}")
    return errors


def validate_evidence(root: Path, registry: Registry) -> list[str]:
    errors: list[str] = []
    try:
        document = EvidenceDocument.model_validate(load_yaml(root / "paper/evidence.yaml"))
    except (OSError, ValueError, ValidationError) as exc:
        return [f"invalid paper/evidence.yaml: {exc}"]
    seen: set[str] = set()
    datasets = registry.datasets()
    runs = registry.runs()
    checkpoints = registry.checkpoints()
    evaluations = registry.evaluations()
    for entry in document.entries:
        if entry.label in seen:
            errors.append(f"duplicate evidence label: {entry.label}")
        seen.add(entry.label)
        references = (
            entry.run_ids or entry.evaluation_ids or entry.dataset_ids or entry.checkpoint_ids
        )
        if not references and not entry.artifact:
            errors.append(f"{entry.label}: evidence has no registered source or artifact")
        for value, index, kind in (
            (entry.run_ids, runs, "run"),
            (entry.evaluation_ids, evaluations, "evaluation"),
            (entry.dataset_ids, datasets, "dataset"),
            (entry.checkpoint_ids, checkpoints, "checkpoint"),
        ):
            for reference in value:
                if reference not in index:
                    errors.append(f"{entry.label}: missing {kind} {reference}")
        if entry.artifact:
            artifact = root / entry.artifact
            if not artifact.is_file():
                errors.append(f"{entry.label}: missing artifact {entry.artifact}")
            elif entry.sha256 and sha256_file(artifact) != entry.sha256:
                errors.append(f"{entry.label}: checksum mismatch for {entry.artifact}")
    return errors


def validate_private_data(root: Path) -> list[str]:
    errors: list[str] = []
    candidates = [
        root / "project.yaml",
        root / "paper/evidence.yaml",
        root / "paper/overleaf.yaml",
    ]
    candidates.extend((root / "registry").glob("*/*.yaml"))
    for path in candidates:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(content):
                errors.append(f"possible private data in {path.relative_to(root)}")
    return errors


def validate_repository(root: Path, check_generated: bool = True) -> list[str]:
    errors: list[str] = []
    try:
        settings = load_project(root)
    except (OSError, ValueError, ValidationError) as exc:
        return [f"invalid project.yaml: {exc}"]
    registry = Registry(root / settings.registry_dir)
    errors.extend(registry.validate_references())
    errors.extend(validate_evidence(root, registry))
    errors.extend(validate_agents(root))
    errors.extend(validate_private_data(root))
    try:
        VenueRecord.model_validate(load_yaml(root / "paper/target/venue.yaml"))
        OverleafConfig.model_validate(load_yaml(root / "paper/overleaf.yaml"))
        manifest = ReleaseManifest.model_validate(load_yaml(root / "releases/current.yaml"))
        from researchctl.release import build_candidate

        expected = build_candidate(root)
        for field in (
            "datasets",
            "runs",
            "checkpoints",
            "evaluations",
            "evidence",
            "unlinked",
        ):
            if getattr(manifest, field) != getattr(expected, field):
                errors.append(f"releases/current.yaml is stale for {field}")
    except (OSError, ValueError, ValidationError) as exc:
        errors.append(f"invalid project control file: {exc}")
    if check_generated:
        generated = root / "docs/generated/README.md"
        if generated.is_file() and generated.read_text(encoding="utf-8") != render_registry_docs(
            registry
        ):
            errors.append("docs/generated/README.md is stale; run researchctl data docs")
        for name, expected_schema in render_schemas().items():
            path = root / "schemas" / name
            if not path.is_file() or path.read_text(encoding="utf-8") != expected_schema:
                errors.append(f"schemas/{name} is stale; run researchctl schema export")
    return errors
