"""Narrow Git helpers used for provenance and release gates."""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: bool
    patch: str
    patch_sha256: str | None


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def state(root: Path) -> GitState:
    commit_result = git(root, "rev-parse", "HEAD", check=False)
    commit = commit_result.stdout.strip() if commit_result.returncode == 0 else "unborn"
    status_result = git(root, "status", "--porcelain", check=False)
    if status_result.returncode != 0:
        return GitState(commit="unborn", dirty=True, patch="", patch_sha256=None)
    dirty = bool(status_result.stdout.strip())
    tracked_patch = git(root, "diff", "--binary", check=False).stdout
    staged_patch = git(root, "diff", "--cached", "--binary", check=False).stdout
    patch = tracked_patch + staged_patch
    digest = hashlib.sha256(patch.encode()).hexdigest() if patch else None
    return GitState(commit=commit, dirty=dirty, patch=patch, patch_sha256=digest)
