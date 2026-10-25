"""Command-line interface for reproducible research project maintenance."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

from researchctl.config import load_project
from researchctl.docs import write_registry_docs
from researchctl.io import atomic_write_yaml, dump_yaml_data, find_project_root, load_yaml
from researchctl.operations import (
    evaluate,
    finalize_run,
    register_dataset,
    submit_slurm,
    train_experiment,
)
from researchctl.paper import (
    bootstrap_overleaf,
    build_paper,
    link_evidence,
    overleaf_status,
    pull_overleaf,
    push_generated,
    push_initial,
)
from researchctl.release import (
    anonymize_release,
    check_release,
    freeze_release,
    publish_release,
    refresh_candidate,
)
from researchctl.schema import TemplateInfo, VenueRecord
from researchctl.schema_export import export_schemas
from researchctl.storage import Registry
from researchctl.validation import validate_repository
from researchctl.venue import install_template, sync_venue

app = typer.Typer(
    help="Maintain traceability from research data to paper and release.",
    no_args_is_help=True,
)
venue_app = typer.Typer(help="Submission target discovery and template management.")
data_app = typer.Typer(help="Dataset registration, verification, and documentation.")
run_app = typer.Typer(help="Tracked training runs and Slurm submission.")
eval_app = typer.Typer(help="Checkpoint evaluation by registry ID.")
paper_app = typer.Typer(help="Paper evidence, build, and Overleaf synchronization.")
release_app = typer.Typer(help="Continuous candidate and formal release lifecycle.")
schema_app = typer.Typer(help="Export machine-readable JSON Schema contracts.")
app.add_typer(venue_app, name="venue")
app.add_typer(data_app, name="data")
app.add_typer(run_app, name="run")
app.add_typer(eval_app, name="eval")
app.add_typer(paper_app, name="paper")
app.add_typer(release_app, name="release")
app.add_typer(schema_app, name="schema")


def _root() -> Path:
    try:
        return find_project_root()
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def _echo_model(value: object) -> None:
    if isinstance(value, BaseModel):
        typer.echo(dump_yaml_data(value))
    else:
        typer.echo(json.dumps(value, indent=2, ensure_ascii=False, default=str))


@app.command("init")
def initialize(
    target: Annotated[str, typer.Option(help="Venue ID such as iclr-2027")] = "iclr-2027",
) -> None:
    """Bind a freshly cloned template to one submission target."""
    root = _root()
    settings = load_project(root)
    registries = Registry(root / settings.registry_dir)
    kinds = ("datasets", "runs", "checkpoints", "evaluations")
    if settings.target != target and any(registries.list(kind) for kind in kinds):
        raise typer.BadParameter("cannot change target after registry records exist")
    updated = settings.model_copy(update={"target": target})
    atomic_write_yaml(root / "project.yaml", updated)
    match = re.fullmatch(r"iclr-(\d{4})", target.lower())
    if match:
        year = int(match.group(1))
        placeholder = VenueRecord(
            id=target,
            name="International Conference on Learning Representations",
            year=year,
            status="awaiting_official_template",
            source_urls=[
                f"https://iclr.cc/Conferences/{year}/AuthorGuidelines",
                f"https://openreview.net/group?id=ICLR.cc/{year}/Conference",
                "https://github.com/ICLR/Master-Template",
            ],
            submission_url=f"https://openreview.net/group?id=ICLR.cc/{year}/Conference",
            template=TemplateInfo(
                advertised_url=(
                    f"https://github.com/ICLR/Master-Template/raw/master/iclr{year}.zip"
                )
            ),
            notes=["Run researchctl venue sync; never use a prior-year template."],
        )
        atomic_write_yaml(root / "paper/target/venue.yaml", placeholder)
    refresh_candidate(root)
    typer.echo(f"Initialized target {target}")


@app.command("validate")
def validate(
    no_generated: Annotated[
        bool, typer.Option("--no-generated", help="Skip generated-doc freshness check")
    ] = False,
) -> None:
    """Validate schemas, references, instructions, privacy, and generated state."""
    errors = validate_repository(_root(), check_generated=not no_generated)
    if errors:
        typer.echo("Validation failed:\n- " + "\n- ".join(errors), err=True)
        raise typer.Exit(1)
    typer.echo("Validation passed")


@schema_app.command("export")
def schema_export() -> None:
    paths = export_schemas(_root())
    typer.echo(f"Exported {len(paths)} schemas")


@venue_app.command("sync")
def venue_sync() -> None:
    root = _root()
    record = sync_venue(root, load_project(root).target)
    _echo_model(record)


@venue_app.command("status")
def venue_status() -> None:
    _echo_model(load_yaml(_root() / "paper/target/venue.yaml"))


@venue_app.command("install-template")
def venue_install_template(
    source_url: Annotated[str | None, typer.Option("--source-url")] = None,
    sha256: Annotated[str | None, typer.Option("--sha256")] = None,
    apply: Annotated[bool, typer.Option("--apply", help="Apply the reviewed diff")] = False,
) -> None:
    report = install_template(_root(), source_url=source_url, expected_sha256=sha256, apply=apply)
    typer.echo(str(report))


def _parse_key_values(values: list[str], label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise typer.BadParameter(f"{label} values use NAME=VALUE: {value}")
        key, item = value.split("=", 1)
        if not key or not item:
            raise typer.BadParameter(f"invalid {label}: {value}")
        parsed[key] = item
    return parsed


@data_app.command("register")
def data_register(
    name: Annotated[str, typer.Option()],
    version: Annotated[str, typer.Option()],
    source: Annotated[str, typer.Option()],
    license_name: Annotated[str, typer.Option("--license")],
    split: Annotated[list[str] | None, typer.Option(help="Repeat NAME=HASH")] = None,
    preprocessing: Annotated[str, typer.Option()] = "none",
    redistributable: Annotated[bool, typer.Option()] = False,
    dvc_path: Annotated[str | None, typer.Option()] = None,
    hf_revision: Annotated[str | None, typer.Option()] = None,
    content_sha256: Annotated[str | None, typer.Option()] = None,
) -> None:
    record = register_dataset(
        _root(),
        name=name,
        version=version,
        source=source,
        license_name=license_name,
        splits=_parse_key_values(split or [], "split"),
        preprocessing=preprocessing,
        redistributable=redistributable,
        dvc_path=dvc_path,
        hf_revision=hf_revision,
        content_sha256=content_sha256,
    )
    _echo_model(record)


@data_app.command("verify")
def data_verify() -> None:
    root = _root()
    registry = Registry(root / load_project(root).registry_dir)
    errors = registry.validate_references()
    for dataset in registry.datasets().values():
        if dataset.dvc_path and not (root / dataset.dvc_path).exists():
            errors.append(f"{dataset.id}: missing DVC pointer/path {dataset.dvc_path}")
    if errors:
        typer.echo("Data verification failed:\n- " + "\n- ".join(errors), err=True)
        raise typer.Exit(1)
    typer.echo(f"Verified {len(registry.datasets())} datasets")


@data_app.command("docs")
def data_docs() -> None:
    root = _root()
    settings = load_project(root)
    path = write_registry_docs(root, Registry(root / settings.registry_dir))
    typer.echo(str(path))


@run_app.command("train")
def run_train(
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    override: Annotated[list[str] | None, typer.Option(help="Hydra-style KEY=VALUE")] = None,
    run_id: Annotated[
        str | None, typer.Option(help="Required for deterministic multi-rank IDs")
    ] = None,
) -> None:
    root = _root()
    settings = load_project(root)
    config_path = (config or root / settings.default_experiment).resolve()
    record = train_experiment(root, config_path, override, run_id)
    if record is not None:
        _echo_model(record)


@run_app.command("finalize")
def run_finalize(identifier: Annotated[str, typer.Argument()]) -> None:
    _echo_model(finalize_run(_root(), identifier))


@run_app.command("submit")
def run_submit(
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    override: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    root = _root()
    settings = load_project(root)
    job_id = submit_slurm(root, (config or root / settings.default_experiment).resolve(), override)
    typer.echo(f"Submitted Slurm job {job_id}")


@eval_app.command("run")
def eval_run(
    checkpoint_id: Annotated[str, typer.Option()],
    dataset_id: Annotated[str, typer.Option()],
    suite: Annotated[str, typer.Option()] = "default",
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    metrics_json: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
) -> None:
    _echo_model(
        evaluate(
            _root(),
            checkpoint_identifier=checkpoint_id,
            dataset_identifier=dataset_id,
            suite=suite,
            config_path=config,
            metrics_json=metrics_json,
        )
    )


@paper_app.command("bootstrap-overleaf")
def paper_bootstrap_overleaf(
    project_id: Annotated[str | None, typer.Option()] = None,
    remote_url: Annotated[str | None, typer.Option(help="Useful for Server Pro or tests")] = None,
) -> None:
    _echo_model(bootstrap_overleaf(_root(), project_id=project_id, remote_url=remote_url))


@paper_app.command("status")
def paper_status() -> None:
    _echo_model(overleaf_status(_root()))


@paper_app.command("pull")
def paper_pull() -> None:
    typer.echo(f"Pulled Overleaf HEAD {pull_overleaf(_root())}")


@paper_app.command("push-initial")
def paper_push_initial() -> None:
    typer.echo(f"Initialized Overleaf at {push_initial(_root())}")


@paper_app.command("push-generated")
def paper_push_generated() -> None:
    typer.echo(f"Pushed generated assets; Overleaf HEAD {push_generated(_root())}")


@paper_app.command("build")
def paper_build() -> None:
    typer.echo(str(build_paper(_root())))


@paper_app.command("link")
def paper_link(
    label: Annotated[str, typer.Option()],
    kind: Annotated[str, typer.Option()],
    source: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    generator: Annotated[str | None, typer.Option()] = None,
    run_id: Annotated[list[str] | None, typer.Option()] = None,
    eval_id: Annotated[list[str] | None, typer.Option()] = None,
    dataset_id: Annotated[list[str] | None, typer.Option()] = None,
    checkpoint_id: Annotated[list[str] | None, typer.Option()] = None,
    note: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    _echo_model(
        link_evidence(
            _root(),
            label=label,
            kind=kind,
            source=source,
            generator=generator,
            run_ids=run_id or [],
            evaluation_ids=eval_id or [],
            dataset_ids=dataset_id or [],
            checkpoint_ids=checkpoint_id or [],
            notes=note,
        )
    )


@release_app.command("refresh")
def release_refresh() -> None:
    _echo_model(refresh_candidate(_root()))


@release_app.command("check")
def release_check(
    strict: Annotated[bool, typer.Option()] = False,
    build: Annotated[bool, typer.Option(help="Compile paper after metadata checks")] = False,
) -> None:
    errors = check_release(_root(), strict=strict, build_paper=build)
    if errors:
        typer.echo("Release checks failed:\n- " + "\n- ".join(errors), err=True)
        raise typer.Exit(1)
    typer.echo("Release checks passed")


@release_app.command("freeze")
def release_freeze(version: Annotated[str, typer.Option()]) -> None:
    typer.echo(str(freeze_release(_root(), version)))


@release_app.command("anonymize")
def release_anonymize(version: Annotated[str, typer.Option()]) -> None:
    typer.echo(str(anonymize_release(_root(), version)))


@release_app.command("publish")
def release_publish(
    version: Annotated[str, typer.Option()],
    dry_run: Annotated[bool, typer.Option("--dry-run/--no-dry-run")] = True,
    confirm: Annotated[bool, typer.Option()] = False,
) -> None:
    _echo_model(publish_release(_root(), version, dry_run=dry_run, confirm=confirm))


if __name__ == "__main__":
    app()
