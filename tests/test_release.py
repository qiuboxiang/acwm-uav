from __future__ import annotations

import zipfile
from pathlib import Path

from researchctl.io import atomic_write_yaml, load_yaml
from researchctl.release import anonymize_release, build_candidate, publication_plan
from researchctl.schema import ProjectSettings


def test_anonymous_bundle_excludes_overleaf_control(project: Path) -> None:
    version = "v0.1.0"
    destination = project / "releases/history" / version
    destination.mkdir(parents=True)
    manifest = build_candidate(project).model_copy(update={"version": version, "status": "frozen"})
    atomic_write_yaml(destination / "manifest.yaml", manifest)
    archive_path = anonymize_release(project, version)
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "paper/overleaf.yaml" not in names
    assert "project.yaml" in names
    assert f"releases/history/{version}/manifest.yaml" in names


def test_publication_plan_keeps_external_targets_explicit(project: Path) -> None:
    version = "v0.2.0"
    destination = project / "releases/history" / version
    destination.mkdir(parents=True)
    atomic_write_yaml(
        destination / "manifest.yaml",
        build_candidate(project).model_copy(update={"version": version, "status": "frozen"}),
    )
    settings_path = project / "project.yaml"
    settings = ProjectSettings.model_validate(load_yaml(settings_path)).model_copy(
        update={
            "github_repository": "example/project",
            "hf_org": "example",
        }
    )
    atomic_write_yaml(settings_path, settings)
    plan = publication_plan(project, version)
    assert plan["github_repository"] == "example/project"
    assert plan["hf_model_repository"] == "example/ai-research-project-model"
    assert plan["hf_dataset_repository"] == "example/ai-research-project-data"
