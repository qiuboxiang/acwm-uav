"""Continuous candidate manifests, release gates, anonymization, and publishing."""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchctl.config import load_project
from researchctl.git import state as git_state
from researchctl.io import (
    atomic_write_text,
    atomic_write_yaml,
    dump_yaml_data,
    load_yaml,
    sha256_file,
    validate_relative,
)
from researchctl.schema import (
    DatasetRecord,
    EvidenceDocument,
    OverleafConfig,
    ReleaseManifest,
)
from researchctl.storage import Registry

SEMVER = re.compile(r"^v\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
ANONYMOUS_EXCLUDES = {
    ".git",
    ".cache",
    ".mlruns",
    ".runs",
    ".venv",
    "dist",
    "artifacts",
    "paper/overleaf.yaml",
    "paper/.overleaf-base.json",
}
IDENTITY_PATTERNS = [
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
    re.compile(r"/home/[A-Za-z0-9._-]+"),
    re.compile(r"git\.overleaf\.com/[0-9a-f]+", re.IGNORECASE),
    re.compile(r"(?i)(token|password|secret)\s*[:=]\s*[^\s\"']{8,}"),
]


def _now() -> datetime:
    return datetime.now(UTC)


def _evidence(root: Path) -> EvidenceDocument:
    return EvidenceDocument.model_validate(load_yaml(root / "paper/evidence.yaml"))


def _registry(root: Path) -> Registry:
    settings = load_project(root)
    return Registry(root / settings.registry_dir)


def _manifest_checksums(root: Path) -> dict[str, str]:
    paths = sorted((root / "registry").glob("*/*.yaml"))
    paths.append(root / "paper/evidence.yaml")
    return {
        path.relative_to(root).as_posix(): sha256_file(path) for path in paths if path.is_file()
    }


def _evidence_closure(
    registry: Registry, evidence: EvidenceDocument
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    selected: dict[str, set[str]] = {
        "datasets": set(),
        "runs": set(),
        "checkpoints": set(),
        "evaluations": set(),
    }
    direct_runs: set[str] = set()
    for entry in evidence.entries:
        direct_runs.update(entry.run_ids)
        selected["runs"].update(entry.run_ids)
        selected["evaluations"].update(entry.evaluation_ids)
        selected["datasets"].update(entry.dataset_ids)
        selected["checkpoints"].update(entry.checkpoint_ids)
    evaluations = registry.evaluations()
    checkpoints = registry.checkpoints()
    runs = registry.runs()
    for identifier in list(selected["evaluations"]):
        evaluation = evaluations.get(identifier)
        if evaluation:
            selected["runs"].add(evaluation.run_id)
            selected["checkpoints"].add(evaluation.checkpoint_id)
            selected["datasets"].add(evaluation.dataset_id)
    for identifier in list(selected["checkpoints"]):
        checkpoint = checkpoints.get(identifier)
        if checkpoint:
            selected["runs"].add(checkpoint.run_id)
    for identifier in list(selected["runs"]):
        run = runs.get(identifier)
        if run:
            selected["datasets"].update(run.dataset_ids)
            if identifier in direct_runs:
                selected["checkpoints"].update(run.checkpoint_ids)
    inventory = {
        "datasets": set(registry.datasets()),
        "runs": set(registry.runs()),
        "checkpoints": set(registry.checkpoints()),
        "evaluations": set(registry.evaluations()),
    }
    linked = {key: sorted(values) for key, values in selected.items()}
    unlinked = {key: sorted(inventory[key] - values) for key, values in selected.items()}
    return linked, unlinked


def build_candidate(root: Path) -> ReleaseManifest:
    registry = _registry(root)
    code = git_state(root)
    overleaf = OverleafConfig.model_validate(load_yaml(root / "paper/overleaf.yaml"))
    evidence = _evidence(root)
    linked, unlinked = _evidence_closure(registry, evidence)
    return ReleaseManifest(
        version="0.0.0-dev",
        status="candidate",
        generated_at=_now(),
        code_commit=code.commit,
        overleaf_head=overleaf.head,
        datasets=linked["datasets"],
        runs=linked["runs"],
        checkpoints=linked["checkpoints"],
        evaluations=linked["evaluations"],
        evidence=sorted(entry.label for entry in evidence.entries),
        checksums=_manifest_checksums(root),
        unlinked=unlinked,
    )


def refresh_candidate(root: Path) -> ReleaseManifest:
    from researchctl.docs import write_registry_docs

    manifest = build_candidate(root)
    atomic_write_yaml(root / "releases/current.yaml", manifest)
    write_registry_docs(root, _registry(root))
    return manifest


def check_release(root: Path, strict: bool = False, build_paper: bool = False) -> list[str]:
    from researchctl.validation import validate_repository

    errors = validate_repository(root)
    registry = _registry(root)
    for dataset in registry.datasets().values():
        if not dataset.license.strip():
            errors.append(f"{dataset.id}: missing license")
        if dataset.redistributable and not (dataset.hf_revision or dataset.dvc_path):
            errors.append(f"{dataset.id}: redistributable data has no artifact location")
    for checkpoint in registry.checkpoints().values():
        payload = root / checkpoint.path
        if payload.is_file() and sha256_file(payload) != checkpoint.sha256:
            errors.append(f"{checkpoint.id}: checkpoint checksum mismatch")
        if strict and not payload.is_file() and not checkpoint.hf_revision:
            errors.append(f"{checkpoint.id}: payload unavailable locally and on Hugging Face")
    if strict:
        state = git_state(root)
        if state.commit == "unborn":
            errors.append("formal release requires at least one Git commit")
        if state.dirty:
            errors.append("formal release requires a clean Git worktree")
        candidate = ReleaseManifest.model_validate(load_yaml(root / "releases/current.yaml"))
        if any(registry.runs()[identifier].dirty for identifier in candidate.runs):
            errors.append("formal release references one or more dirty-code runs")
        for field in ("datasets", "runs", "checkpoints", "evaluations", "evidence"):
            if not getattr(candidate, field):
                errors.append(f"formal release requires at least one {field} entry")
        overleaf = OverleafConfig.model_validate(load_yaml(root / "paper/overleaf.yaml"))
        if not overleaf.head:
            errors.append("formal release requires a synchronized Overleaf HEAD")
    if build_paper and not errors:
        from researchctl.paper import build_paper as compile_paper

        try:
            compile_paper(root)
        except Exception as exc:  # subprocess failures are summarized for release output
            errors.append(f"paper build failed: {exc}")
    return sorted(set(errors))


def _model_card(root: Path, manifest: ReleaseManifest) -> str:
    return (
        "# Model card\n\n"
        f"Release: `{manifest.version}`\n\n"
        "## Provenance\n\n"
        f"- Runs: {', '.join(manifest.runs)}\n"
        f"- Checkpoints: {', '.join(manifest.checkpoints)}\n"
        f"- Evaluations: {', '.join(manifest.evaluations)}\n"
        "\nSee `manifest.yaml` for immutable checksums and dataset references.\n"
    )


def _data_card(root: Path, manifest: ReleaseManifest) -> str:
    registry = _registry(root)
    lines = ["# Dataset card\n", f"Release: `{manifest.version}`\n"]
    for identifier in manifest.datasets:
        dataset = registry.load("datasets", identifier, DatasetRecord)
        lines.extend(
            [
                f"\n## {identifier}\n",
                f"- Source: {dataset.source}\n",
                f"- License: {dataset.license}\n",
                f"- Redistributable: {dataset.redistributable}\n",
                f"- Preprocessing: {dataset.preprocessing}\n",
            ]
        )
    return "".join(lines)


def freeze_release(root: Path, version: str) -> Path:
    if not SEMVER.match(version):
        raise ValueError("version must be semantic and start with v, for example v1.0.0")
    errors = check_release(root, strict=True, build_paper=True)
    if errors:
        raise ValueError("release checks failed:\n- " + "\n- ".join(errors))
    destination = root / "releases/history" / version
    if destination.exists():
        raise FileExistsError(f"release already exists and is immutable: {version}")
    candidate = build_candidate(root)
    settings = load_project(root)
    hf_revisions: dict[str, str] = {}
    model_repo = settings.hf_model_repository or (
        f"{settings.hf_org}/{settings.name}-model" if settings.hf_org else None
    )
    dataset_repo = settings.hf_dataset_repository or (
        f"{settings.hf_org}/{settings.name}-data" if settings.hf_org else None
    )
    if model_repo:
        hf_revisions["model"] = f"{model_repo}@{version}"
    if dataset_repo:
        hf_revisions["dataset"] = f"{dataset_repo}@{version}"
    frozen = candidate.model_copy(
        update={"version": version, "status": "frozen", "hf_revisions": hf_revisions}
    )
    destination.mkdir(parents=True)
    atomic_write_text(
        destination / "AGENTS.md",
        "# Frozen Release Instructions\n\n"
        "- This directory is immutable. Create a new semantic version for corrections.\n"
        "- Preserve manifests, cards, checksums, and publication records together.\n",
    )
    atomic_write_yaml(destination / "manifest.yaml", frozen)
    atomic_write_text(destination / "MODEL_CARD.md", _model_card(root, frozen))
    atomic_write_text(destination / "DATASET_CARD.md", _data_card(root, frozen))
    atomic_write_text(
        destination / "REPRODUCE.md",
        "# Reproduction\n\n"
        "```bash\nuv sync --all-extras\ndvc pull\nresearchctl validate\n"
        "researchctl release check --strict\n```\n",
    )
    atomic_write_text(
        destination / "checksums.sha256",
        "".join(f"{digest}  {path}\n" for path, digest in frozen.checksums.items()),
    )
    return destination


def _anonymous_paths(root: Path) -> list[Path]:
    roots = ["src", "configs", "registry", "tests", "docs", "paper"]
    files = [
        root / "README.md",
        root / "pyproject.toml",
        root / "project.yaml",
        root / "dvc.yaml",
    ]
    for directory_name in roots:
        directory = root / directory_name
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    selected = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if any(relative == item or relative.startswith(item + "/") for item in ANONYMOUS_EXCLUDES):
            continue
        if path.suffix in {
            ".pt",
            ".ckpt",
            ".pdf",
            ".log",
            ".aux",
            ".bbl",
            ".blg",
            ".fdb_latexmk",
            ".fls",
            ".out",
        } or path.name.endswith(".synctex.gz"):
            continue
        selected.append(path)
    return sorted(set(selected))


def anonymize_release(root: Path, version: str) -> Path:
    manifest_path = root / "releases/history" / version / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"freeze release first: {version}")
    offenders: list[str] = []
    paths = _anonymous_paths(root)
    for path in paths:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(content) for pattern in IDENTITY_PATTERNS):
            offenders.append(path.relative_to(root).as_posix())
    if offenders:
        raise ValueError("identity scan failed:\n- " + "\n- ".join(offenders))
    output = root / "dist" / f"anonymous-{version}.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            relative = path.relative_to(root).as_posix()
            if relative == "project.yaml":
                project = load_project(root).model_copy(
                    update={
                        "github_repository": None,
                        "hf_org": None,
                        "hf_model_repository": None,
                        "hf_dataset_repository": None,
                    }
                )
                archive.writestr(relative, dump_yaml_data(project))
            else:
                archive.write(path, relative)
        archive.write(manifest_path, f"releases/history/{version}/manifest.yaml")
    return output


def publication_plan(root: Path, version: str) -> dict[str, Any]:
    settings = load_project(root)
    registry = _registry(root)
    manifest = ReleaseManifest.model_validate(
        load_yaml(root / "releases/history" / version / "manifest.yaml")
    )
    datasets = [registry.load("datasets", value, DatasetRecord) for value in manifest.datasets]
    return {
        "version": version,
        "github_repository": settings.github_repository,
        "hf_model_repository": settings.hf_model_repository
        or (f"{settings.hf_org}/{settings.name}-model" if settings.hf_org else None),
        "hf_dataset_repository": settings.hf_dataset_repository
        or (f"{settings.hf_org}/{settings.name}-data" if settings.hf_org else None),
        "checkpoints": manifest.checkpoints,
        "redistributable_datasets": [item.id for item in datasets if item.redistributable],
        "metadata_only_datasets": [item.id for item in datasets if not item.redistributable],
        "visibility": settings.release_visibility,
    }


def _dataset_payload(root: Path, dataset: DatasetRecord) -> Path | None:
    if not dataset.dvc_path:
        return None
    validate_relative(dataset.dvc_path)
    pointer_or_payload = root / dataset.dvc_path
    if pointer_or_payload.suffix != ".dvc":
        return pointer_or_payload if pointer_or_payload.exists() else None
    if not pointer_or_payload.is_file():
        return None
    pointer = load_yaml(pointer_or_payload)
    outputs = pointer.get("outs", [])
    if not isinstance(outputs, list) or not outputs:
        return None
    first = outputs[0]
    if not isinstance(first, dict) or not isinstance(first.get("path"), str):
        return None
    payload = (pointer_or_payload.parent / first["path"]).resolve()
    try:
        payload.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"DVC payload escapes project: {dataset.id}") from exc
    return payload if payload.exists() else None


def publish_release(root: Path, version: str, *, dry_run: bool, confirm: bool) -> dict[str, Any]:
    plan = publication_plan(root, version)
    if dry_run:
        return plan
    if not confirm:
        raise ValueError("external publication requires --confirm; use --dry-run to inspect")
    if git_state(root).dirty:
        raise ValueError("commit the frozen release files before publishing")
    if not plan["github_repository"]:
        raise ValueError("project.yaml must define github_repository")
    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError("GitHub CLI 'gh' is required for publication")
    subprocess.run(["git", "tag", "--annotate", version, "-m", version], cwd=root, check=True)
    subprocess.run(["git", "push", "origin", version], cwd=root, check=True)
    subprocess.run(
        [
            gh,
            "release",
            "create",
            version,
            f"releases/history/{version}/manifest.yaml",
            f"releases/history/{version}/checksums.sha256",
            "--repo",
            str(plan["github_repository"]),
            "--title",
            version,
            "--generate-notes",
        ],
        cwd=root,
        check=True,
    )
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("install the research extra for Hugging Face publishing") from exc
    api = HfApi()
    private = load_project(root).release_visibility == "private"
    registry = _registry(root)
    model_repo = plan["hf_model_repository"]
    dataset_repo = plan["hf_dataset_repository"]
    if model_repo:
        api.create_repo(model_repo, repo_type="model", private=private, exist_ok=True)
        api.create_branch(model_repo, branch=version, repo_type="model", exist_ok=True)
        for identifier in plan["checkpoints"]:
            checkpoint = registry.checkpoints()[identifier]
            payload = root / checkpoint.path
            if payload.is_file():
                api.upload_file(
                    path_or_fileobj=payload,
                    path_in_repo=payload.name,
                    repo_id=model_repo,
                    repo_type="model",
                    revision=version,
                )
    if dataset_repo:
        api.create_repo(dataset_repo, repo_type="dataset", private=private, exist_ok=True)
        api.create_branch(dataset_repo, branch=version, repo_type="dataset", exist_ok=True)
        data_card = root / "releases/history" / version / "DATASET_CARD.md"
        api.upload_file(
            path_or_fileobj=data_card,
            path_in_repo="README.md",
            repo_id=dataset_repo,
            repo_type="dataset",
            revision=version,
        )
        for identifier in plan["redistributable_datasets"]:
            dataset = registry.datasets()[identifier]
            dataset_payload = _dataset_payload(root, dataset)
            if dataset_payload is None:
                raise FileNotFoundError(
                    f"redistributable dataset payload unavailable for publication: {identifier}"
                )
            if dataset_payload.is_dir():
                api.upload_folder(
                    folder_path=dataset_payload,
                    path_in_repo=f"data/{dataset.name}/{dataset.version}",
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    revision=version,
                )
            else:
                api.upload_file(
                    path_or_fileobj=dataset_payload,
                    path_in_repo=(f"data/{dataset.name}/{dataset.version}/{dataset_payload.name}"),
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    revision=version,
                )
    atomic_write_yaml(
        root / "releases/history" / version / "publication.yaml",
        {"schema_version": 1, "published_at": _now().isoformat(), **plan},
    )
    return plan
