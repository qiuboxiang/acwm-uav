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
    # Gitignored local assets (datasets, simulator scenes, vendored code, experiment outputs) can be tens of GB;
    # copying them into every test's tmp dir fills the disk. Skip them at their known locations only.
    local_assets = {source: {"data", "scenes", "third_party", "IndoorUAV-Agent"}, source / "src" / "research": {"exploration_v5", "exploration_dev"}}

    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = set(ignored(directory, names))
        skipped |= local_assets.get(Path(directory), set()) & set(names)
        if Path(directory) == source / "src" / "research" / "exploration":  # gitignored experiment outputs
            skipped |= {n for n in names if n.startswith(("results", "logs"))}
        return skipped

    shutil.copytree(source, destination, ignore=ignore)
    refresh_candidate(destination)
    return destination
