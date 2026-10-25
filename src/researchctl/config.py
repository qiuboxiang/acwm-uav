"""Project and experiment configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from researchctl.io import load_yaml
from researchctl.schema import ProjectSettings


def load_project(root: Path) -> ProjectSettings:
    return ProjectSettings.model_validate(load_yaml(root / "project.yaml"))


def load_experiment(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("experiment config must be a mapping")
    return {str(key): value for key, value in resolved.items()}
