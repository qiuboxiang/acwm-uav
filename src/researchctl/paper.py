"""Paper evidence, compilation, and guarded Overleaf Git Bridge synchronization."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import webbrowser
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from urllib.parse import parse_qs, urlsplit

from researchctl.config import load_project
from researchctl.io import (
    atomic_write_text,
    atomic_write_yaml,
    load_yaml,
    relative_path,
    sha256_file,
)
from researchctl.schema import EvidenceDocument, EvidenceEntry, OverleafConfig
from researchctl.storage import Registry

CONTROL_FILES = {
    "AGENTS.md",
    "evidence.yaml",
    "overleaf.yaml",
    ".overleaf-base.json",
}
BUILD_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".pdf",
    ".synctex.gz",
}


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _syncable(relative: Path) -> bool:
    value = relative.as_posix()
    if (
        value in CONTROL_FILES
        or relative.name == "AGENTS.md"
        or value.startswith("target/")
        or value.startswith("sync-conflicts/")
    ):
        return False
    if any(value.endswith(suffix) for suffix in BUILD_SUFFIXES):
        return False
    return not any(part.startswith(".") for part in relative.parts)


def _files(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not directory.exists():
        return result
    for path in directory.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            relative = path.relative_to(directory)
            if _syncable(relative):
                result[relative.as_posix()] = path
    return result


def _hashes(directory: Path) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in _files(directory).items()}


def _load_base(root: Path) -> dict[str, object]:
    path = root / "paper/.overleaf-base.json"
    if not path.is_file():
        return {"head": None, "files": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("files", {}), dict):
        raise ValueError("paper/.overleaf-base.json is invalid")
    return raw


def _write_base(root: Path, head: str, files: dict[str, str]) -> None:
    atomic_write_text(
        root / "paper/.overleaf-base.json",
        json.dumps({"head": head, "files": files}, indent=2, sort_keys=True) + "\n",
    )


def _config(root: Path) -> OverleafConfig:
    return OverleafConfig.model_validate(load_yaml(root / "paper/overleaf.yaml"))


def _mirror_path(root: Path) -> Path:
    return root / ".cache/overleaf/mirror"


def _ensure_mirror(root: Path) -> tuple[Path, OverleafConfig]:
    config = _config(root)
    if not config.remote_url:
        raise ValueError("Overleaf is not configured; run paper bootstrap-overleaf")
    mirror = _mirror_path(root)
    if not (mirror / ".git").is_dir():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["git", "clone", config.remote_url, str(mirror)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"unable to clone Overleaf project: {result.stderr.strip()}")
    return mirror, config


def bootstrap_overleaf(
    root: Path, *, project_id: str | None, remote_url: str | None = None
) -> OverleafConfig:
    if not project_id and not remote_url:
        webbrowser.open("https://www.overleaf.com/project")
        raise ValueError("create a blank Premium Overleaf project, then rerun with --project-id")
    identifier = project_id or "custom-remote"
    url = remote_url or f"https://git.overleaf.com/{identifier}"
    parsed = urlsplit(url)
    if parsed.password:
        raise ValueError("remote URL must not embed an Overleaf token or password")
    sensitive_query = {"token", "password", "secret"} & set(parse_qs(parsed.query))
    if sensitive_query:
        raise ValueError("remote URL must not contain credential query parameters")
    config = OverleafConfig(project_id=project_id, remote_url=url)
    atomic_write_yaml(root / "paper/overleaf.yaml", config)
    return config


def overleaf_status(root: Path) -> dict[str, object]:
    config = _config(root)
    base = _load_base(root)
    local = _hashes(root / "paper")
    baseline = base.get("files", {})
    assert isinstance(baseline, dict)
    changed = sorted(
        name for name in set(local) | set(baseline) if local.get(name) != baseline.get(name)
    )
    return {
        "configured": bool(config.remote_url),
        "project_id": config.project_id,
        "head": config.head,
        "local_changes_since_pull": changed,
    }


def _checkout_remote(mirror: Path, branch: str) -> str:
    fetch = _git(mirror, "fetch", "origin", branch, check=False)
    if fetch.returncode != 0:
        raise RuntimeError(f"Overleaf fetch failed: {fetch.stderr.strip()}")
    checkout = _git(mirror, "checkout", "-B", branch, f"origin/{branch}", check=False)
    if checkout.returncode != 0:
        raise RuntimeError(f"Overleaf checkout failed: {checkout.stderr.strip()}")
    return _git(mirror, "rev-parse", "HEAD").stdout.strip()


def pull_overleaf(root: Path) -> str:
    mirror, config = _ensure_mirror(root)
    head = _checkout_remote(mirror, config.branch)
    paper_root = root / "paper"
    base = _load_base(root)
    baseline_raw = base.get("files", {})
    assert isinstance(baseline_raw, dict)
    baseline = {str(key): str(value) for key, value in baseline_raw.items()}
    local = _hashes(paper_root)
    remote = _hashes(mirror)
    has_baseline = bool(base.get("head"))
    conflicts: list[str] = []
    for name in sorted(set(baseline) | set(local) | set(remote)):
        previous = baseline.get(name)
        local_changed = has_baseline and local.get(name) != previous
        remote_changed = remote.get(name) != previous
        if local_changed and remote_changed and local.get(name) != remote.get(name):
            conflicts.append(name)
    if conflicts:
        report = root / "paper/sync-conflicts/overleaf.md"
        atomic_write_text(
            report,
            "# Overleaf synchronization conflicts\n\n"
            + "".join(f"- `{name}` changed locally and remotely.\n" for name in conflicts),
        )
        raise RuntimeError(f"Overleaf conflicts detected; inspect {report.relative_to(root)}")
    remote_files = _files(mirror)
    for name in sorted(set(baseline) | set(remote)):
        previous = baseline.get(name)
        if remote.get(name) == previous:
            continue
        destination = paper_root / name
        source = remote_files.get(name)
        if source is None:
            if destination.is_file():
                destination.unlink()
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _write_base(root, head, remote)
    atomic_write_yaml(root / "paper/overleaf.yaml", config.model_copy(update={"head": head}))
    return head


def _configure_mirror_identity(mirror: Path) -> None:
    _git(mirror, "config", "user.name", "researchctl")
    _git(mirror, "config", "user.email", "researchctl@localhost")


def push_initial(root: Path) -> str:
    mirror, config = _ensure_mirror(root)
    current = _git(mirror, "rev-parse", "HEAD", check=False)
    if current.returncode == 0 and _files(mirror):
        raise ValueError(
            "Overleaf project is not empty; delete the blank main.tex or pull it before bootstrap"
        )
    _configure_mirror_identity(mirror)
    for name, source in _files(root / "paper").items():
        destination = mirror / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    _git(mirror, "add", "--all")
    _git(mirror, "commit", "-m", "Initialize paper from research template")
    _git(mirror, "push", "origin", f"HEAD:{config.branch}")
    return pull_overleaf(root)


def push_generated(root: Path) -> str:
    pull_overleaf(root)
    mirror, config = _ensure_mirror(root)
    _configure_mirror_identity(mirror)
    generated = root / "paper/generated"
    mirror_generated = mirror / "generated"
    mirror_generated.mkdir(parents=True, exist_ok=True)
    local_names = {
        path.relative_to(generated).as_posix()
        for path in generated.rglob("*")
        if path.is_file() and path.name != "AGENTS.md"
    }
    for path in list(mirror_generated.rglob("*")):
        if path.is_file() and path.relative_to(mirror_generated).as_posix() not in local_names:
            path.unlink()
    for path in generated.rglob("*"):
        if path.is_file() and path.name != "AGENTS.md":
            destination = mirror_generated / path.relative_to(generated)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    _git(mirror, "add", "--all")
    diff = _git(mirror, "diff", "--cached", "--quiet", check=False)
    if diff.returncode != 0:
        _git(mirror, "commit", "-m", f"Update generated evidence {datetime.now(UTC).isoformat()}")
        _git(mirror, "push", "origin", config.branch)
    return pull_overleaf(root)


def build_paper(root: Path) -> Path:
    settings = load_project(root)
    main = root / settings.paper_main
    if not main.is_file():
        raise FileNotFoundError(f"paper main file not found: {settings.paper_main}")
    latexmk = shutil.which("latexmk")
    if latexmk is None:
        raise RuntimeError(
            "latexmk is unavailable; install TeX Live or use docker/latex.Dockerfile"
        )
    result = subprocess.run(
        [latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", main.name],
        cwd=main.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-30:])
        raise RuntimeError(tail)
    return main.with_suffix(".pdf")


def link_evidence(
    root: Path,
    *,
    label: str,
    kind: str,
    source: Path | None,
    generator: str | None,
    run_ids: list[str],
    evaluation_ids: list[str],
    dataset_ids: list[str],
    checkpoint_ids: list[str],
    notes: list[str] | None = None,
) -> EvidenceEntry:
    registry = Registry(root / load_project(root).registry_dir)
    for value in run_ids:
        registry.load("runs", value)
    for value in evaluation_ids:
        registry.load("evaluations", value)
    for value in dataset_ids:
        registry.load("datasets", value)
    for value in checkpoint_ids:
        registry.load("checkpoints", value)
    if kind not in {"figure", "table", "claim"}:
        raise ValueError("kind must be figure, table, or claim")
    evidence_kind = cast(Literal["figure", "table", "claim"], kind)
    if kind != "claim" and source is None:
        raise ValueError("figure and table evidence require --source")
    artifact: str | None = None
    checksum: str | None = None
    if source is not None:
        if not source.is_file():
            raise FileNotFoundError(source)
        safe_label = re.sub(r"[^a-zA-Z0-9._-]", "-", label)
        destination = root / "paper/generated" / f"{safe_label}{source.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        artifact = relative_path(root, destination)
        checksum = sha256_file(destination)
    document_path = root / "paper/evidence.yaml"
    document = EvidenceDocument.model_validate(load_yaml(document_path))
    if any(entry.label == label for entry in document.entries):
        raise ValueError(f"evidence label already exists and is immutable: {label}")
    entry = EvidenceEntry(
        label=label,
        kind=evidence_kind,
        artifact=artifact,
        sha256=checksum,
        generator=generator,
        run_ids=run_ids,
        evaluation_ids=evaluation_ids,
        dataset_ids=dataset_ids,
        checkpoint_ids=checkpoint_ids,
        notes=notes or [],
    )
    atomic_write_yaml(
        document_path, document.model_copy(update={"entries": [*document.entries, entry]})
    )
    from researchctl.release import refresh_candidate

    refresh_candidate(root)
    return entry
