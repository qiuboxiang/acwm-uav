"""Dataset, run, checkpoint, and evaluation transactions."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from researchctl.config import load_experiment, load_project
from researchctl.git import state as git_state
from researchctl.ids import checkpoint_id, dataset_id, evaluation_id, filename_for_id, run_id
from researchctl.io import atomic_write_text, relative_path, sha256_file
from researchctl.schema import CheckpointRecord, DatasetRecord, EvaluationRecord, RunRecord
from researchctl.storage import Registry


def _now() -> datetime:
    return datetime.now(UTC)


def _registry(root: Path) -> Registry:
    settings = load_project(root)
    return Registry(root / settings.registry_dir)


def _refresh(root: Path) -> None:
    from researchctl.release import refresh_candidate

    refresh_candidate(root)


def register_dataset(
    root: Path,
    *,
    name: str,
    version: str,
    source: str,
    license_name: str,
    splits: dict[str, str],
    preprocessing: str,
    redistributable: bool = False,
    dvc_path: str | None = None,
    hf_revision: str | None = None,
    content_sha256: str | None = None,
) -> DatasetRecord:
    created = _now()
    record = DatasetRecord(
        id=dataset_id(name, version),
        status="finalized",
        created_at=created,
        finalized_at=created,
        name=name,
        version=version,
        source=source,
        license=license_name,
        redistributable=redistributable,
        splits=splits,
        preprocessing=preprocessing,
        dvc_path=dvc_path,
        hf_revision=hf_revision,
        content_sha256=content_sha256,
    )
    _registry(root).write("datasets", record)
    _refresh(root)
    return record


def _mlflow_context(root: Path, run_name: str, config: dict[str, Any]) -> tuple[Any, str | None]:
    try:
        import mlflow
    except ImportError:
        return nullcontext(), None
    settings = load_project(root)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
    if not tracking_uri.startswith(("http://", "https://", "file:")):
        tracking_uri = (root / tracking_uri).resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    context = mlflow.start_run(run_name=run_name)
    active = mlflow.active_run()
    run_identifier = active.info.run_id if active is not None else None
    flat: dict[str, str] = {}
    for section, values in config.items():
        if isinstance(values, dict):
            flat.update({f"{section}.{key}": str(value) for key, value in values.items()})
        else:
            flat[section] = str(values)
    mlflow.log_params(flat)
    return context, run_identifier


def _try_dvc_add(root: Path, payload: Path) -> str | None:
    executable = shutil.which("dvc")
    if executable is None:
        return None
    result = subprocess.run(
        [executable, "add", relative_path(root, payload)],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc add failed for {payload}: {result.stderr.strip()}")
    dvc_file = payload.with_name(payload.name + ".dvc")
    return relative_path(root, dvc_file) if dvc_file.exists() else None


def train_experiment(
    root: Path,
    config_path: Path,
    overrides: list[str] | None = None,
    explicit_run_id: str | None = None,
) -> RunRecord | None:
    config = load_experiment(config_path, overrides)
    name = str(config.get("name", config_path.stem))
    code = git_state(root)
    identifier = explicit_run_id or os.environ.get("RESEARCH_RUN_ID") or run_id(name, code.commit)
    rank = int(os.environ.get("RANK", "0"))
    registry = _registry(root)
    dataset_reference = str(config["data"]["dataset_id"])
    if dataset_reference not in registry.datasets():
        raise ValueError(
            f"dataset is not registered: {dataset_reference}; run researchctl data register"
        )
    safe_run = filename_for_id(identifier).removesuffix(".yaml")
    output_dir = root / "artifacts/checkpoints" / safe_run
    patch_path = root / ".runs" / safe_run / "dirty.patch"
    if rank == 0 and code.patch:
        atomic_write_text(patch_path, code.patch)
    created = _now()
    draft = RunRecord(
        id=identifier,
        status="draft",
        created_at=created,
        name=name,
        command=list(sys.argv),
        config_source=relative_path(root, config_path),
        resolved_config=config,
        overrides=overrides or [],
        code_commit=code.commit,
        dirty=code.dirty,
        dirty_patch_sha256=code.patch_sha256,
        python_version=platform.python_version(),
        platform=platform.platform(),
        seed=int(config["seed"]),
        launcher=dict(config.get("launcher", {"kind": "local"})),
        dataset_ids=[dataset_reference],
        output_dir=relative_path(root, output_dir),
    )
    if rank == 0:
        registry.write("runs", draft)

    try:
        from research.pipelines.train import run_training

        context, mlflow_run_id = (
            _mlflow_context(root, identifier, config)
            if rank == 0
            else (
                nullcontext(),
                None,
            )
        )
        with context:
            result = run_training(config, output_dir)
            if rank == 0:
                try:
                    import mlflow

                    mlflow.log_metrics(result.metrics)
                except ImportError:
                    pass
    except Exception:
        if rank == 0:
            failed = draft.model_copy(update={"status": "failed", "finalized_at": _now()})
            registry.write("runs", failed)
        raise

    if rank != 0:
        return None
    checkpoints: list[str] = []
    metric_name = str(config["trainer"].get("selection_metric", "val_mse"))
    metric_value = result.metrics.get(metric_name)
    for alias, path in result.checkpoints.items():
        checksum = sha256_file(path)
        checkpoint_record = CheckpointRecord(
            id=checkpoint_id(identifier, alias, checksum),
            status="finalized",
            created_at=_now(),
            finalized_at=_now(),
            run_id=identifier,
            alias=alias,
            format="pytorch-state-dict",
            path=relative_path(root, path),
            dvc_path=_try_dvc_add(root, path),
            sha256=checksum,
            size_bytes=path.stat().st_size,
            selection_metric=metric_name if alias == "best" else None,
            selection_value=metric_value if alias == "best" else None,
        )
        registry.write("checkpoints", checkpoint_record)
        checkpoints.append(checkpoint_record.id)
    finalized = draft.model_copy(
        update={
            "status": "finalized",
            "finalized_at": _now(),
            "metrics": result.metrics,
            "checkpoint_ids": checkpoints,
            "mlflow_run_id": mlflow_run_id,
        }
    )
    registry.write("runs", finalized)
    _refresh(root)
    return finalized


def finalize_run(root: Path, identifier: str) -> RunRecord:
    registry = _registry(root)
    record = registry.load("runs", identifier, RunRecord)
    if record.status == "finalized":
        return record
    finalized = record.model_copy(update={"status": "finalized", "finalized_at": _now()})
    registry.write("runs", finalized)
    _refresh(root)
    return finalized


def evaluate(
    root: Path,
    *,
    checkpoint_identifier: str,
    dataset_identifier: str,
    suite: str,
    config_path: Path | None = None,
    metrics_json: Path | None = None,
) -> EvaluationRecord:
    registry = _registry(root)
    checkpoint = registry.load("checkpoints", checkpoint_identifier, CheckpointRecord)
    run = registry.load("runs", checkpoint.run_id, RunRecord)
    registry.load("datasets", dataset_identifier, DatasetRecord)
    eval_config: dict[str, Any] = {"suite": suite}
    if config_path is not None:
        eval_config = load_experiment(config_path)
    if metrics_json is not None:
        raw_metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        if not isinstance(raw_metrics, dict):
            raise ValueError("metrics JSON must contain an object")
        metrics = {str(key): float(value) for key, value in raw_metrics.items()}
    else:
        from research.pipelines.evaluate import evaluate_checkpoint

        checkpoint_path = root / checkpoint.path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"checkpoint payload missing: {checkpoint.path}; run dvc pull first"
            )
        config_data = dict(run.resolved_config["data"])
        config_data["dataset_id"] = dataset_identifier
        result = evaluate_checkpoint(
            dict(run.resolved_config["model"]), config_data, checkpoint_path
        )
        metrics = result.metrics
    identifier = evaluation_id(run.id, suite)
    created = _now()
    record = EvaluationRecord(
        id=identifier,
        status="finalized",
        created_at=created,
        finalized_at=created,
        run_id=run.id,
        checkpoint_id=checkpoint.id,
        dataset_id=dataset_identifier,
        suite=suite,
        config=eval_config,
        metrics=metrics,
    )
    registry.write("evaluations", record)
    _refresh(root)
    return record


def submit_slurm(root: Path, config_path: Path, overrides: list[str] | None = None) -> str:
    try:
        import submitit
    except ImportError as exc:
        raise RuntimeError("install the research extra to use Slurm: uv sync --all-extras") from exc
    config = load_experiment(config_path, overrides)
    launcher = dict(config.get("launcher", {}))
    executor = submitit.AutoExecutor(folder=str(root / ".runs/submitit"))
    executor.update_parameters(
        timeout_min=int(launcher.get("timeout_min", 60)),
        nodes=int(launcher.get("nodes", 1)),
        gpus_per_node=int(launcher.get("gpus_per_node", 1)),
        cpus_per_task=int(launcher.get("cpus_per_task", 4)),
        mem_gb=int(launcher.get("mem_gb", 16)),
        slurm_partition=launcher.get("partition"),
        slurm_account=launcher.get("account"),
    )
    job = executor.submit(train_experiment, root, config_path, overrides)
    return str(job.job_id)
