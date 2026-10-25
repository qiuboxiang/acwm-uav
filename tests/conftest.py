from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from researchctl.release import refresh_candidate


@pytest.fixture
def project(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    destination = tmp_path / "project"
    ignored = shutil.ignore_patterns(
        ".git",
        ".venv",
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".mlruns",
        ".runs",
        "__pycache__",
        "*.egg-info",
        "build",
        "dist",
    )
    shutil.copytree(source, destination, ignore=ignored)
    refresh_candidate(destination)
    return destination
