from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from researchctl.paper import bootstrap_overleaf, pull_overleaf, push_generated


def _git(path: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True)


def _remote(tmp_path: Path, content: str = "remote version\n") -> tuple[Path, Path]:
    remote = tmp_path / "overleaf.git"
    subprocess.run(
        ["git", "init", "--bare", "--initial-branch=master", str(remote)],
        check=True,
        capture_output=True,
    )
    seed = tmp_path / "seed"
    subprocess.run(
        ["git", "init", "--initial-branch=master", str(seed)], check=True, capture_output=True
    )
    _git(seed, "config", "user.name", "test")
    _git(seed, "config", "user.email", "test@example.invalid")
    (seed / "main.tex").write_text(content, encoding="utf-8")
    _git(seed, "add", "main.tex")
    _git(seed, "commit", "-m", "seed")
    _git(seed, "remote", "add", "origin", str(remote))
    _git(seed, "push", "origin", "master")
    return remote, seed


def test_first_pull_treats_overleaf_as_authority(project: Path, tmp_path: Path) -> None:
    remote, _ = _remote(tmp_path)
    bootstrap_overleaf(project, project_id="test", remote_url=str(remote))
    head = pull_overleaf(project)
    assert len(head) == 40
    assert (project / "paper/main.tex").read_text(encoding="utf-8") == "remote version\n"


def test_two_sided_change_stops_without_overwrite(project: Path, tmp_path: Path) -> None:
    remote, seed = _remote(tmp_path)
    bootstrap_overleaf(project, project_id="test", remote_url=str(remote))
    pull_overleaf(project)
    local = project / "paper/main.tex"
    local.write_text("local edit\n", encoding="utf-8")
    (seed / "main.tex").write_text("remote edit\n", encoding="utf-8")
    _git(seed, "add", "main.tex")
    _git(seed, "commit", "-m", "remote edit")
    _git(seed, "push", "origin", "master")
    with pytest.raises(RuntimeError, match="conflicts"):
        pull_overleaf(project)
    assert local.read_text(encoding="utf-8") == "local edit\n"
    assert (project / "paper/sync-conflicts/overleaf.md").is_file()


def test_generated_assets_push_after_pull(project: Path, tmp_path: Path) -> None:
    remote, _ = _remote(tmp_path)
    bootstrap_overleaf(project, project_id="test", remote_url=str(remote))
    pull_overleaf(project)
    (project / "paper/generated/result.txt").write_text("result\n", encoding="utf-8")
    push_generated(project)
    clone = tmp_path / "verification"
    subprocess.run(["git", "clone", str(remote), str(clone)], check=True, capture_output=True)
    assert (clone / "generated/result.txt").read_text(encoding="utf-8") == "result\n"


def test_bootstrap_rejects_embedded_credentials(project: Path) -> None:
    with pytest.raises(ValueError, match="must not embed"):
        bootstrap_overleaf(
            project,
            project_id="test",
            remote_url="https://git:super-secret@git.overleaf.com/test",
        )
