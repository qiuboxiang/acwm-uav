"""Stable identifier helpers."""

from __future__ import annotations

import re
from datetime import UTC, datetime


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-")
    if not value:
        raise ValueError("value cannot produce an empty slug")
    return value


def dataset_id(name: str, version: str) -> str:
    return f"dataset:{slugify(name)}@{version}"


def run_id(name: str, git_sha: str, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"run:{timestamp}-{slugify(name)}-{git_sha[:7]}"


def checkpoint_id(run: str, alias: str, sha256: str) -> str:
    return f"ckpt:{run.removeprefix('run:')}/{slugify(alias)}-{sha256[:8]}"


def evaluation_id(run: str, suite: str, now: datetime | None = None) -> str:
    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"eval:{run.removeprefix('run:')}/{slugify(suite)}-{timestamp}"


def filename_for_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "__", value) + ".yaml"
