"""Validated, versioned data contracts shared by all repository workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Contract(BaseModel):
    """Base contract with strict forward-compatibility behavior."""

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1


class ProjectSettings(Contract):
    name: str
    target: str
    default_experiment: str
    paper_main: str
    registry_dir: str = "registry"
    artifact_dir: str = "artifacts"
    mlflow_tracking_uri: str = ".mlruns"
    dvc_remote: str | None = None
    github_repository: str | None = None
    hf_org: str | None = None
    hf_model_repository: str | None = None
    hf_dataset_repository: str | None = None
    release_visibility: Literal["private", "public"] = "private"


class Record(Contract):
    id: str
    status: Literal["draft", "finalized", "failed"] = "finalized"
    created_at: datetime
    finalized_at: datetime | None = None
    supersedes: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def stable_id(cls, value: str) -> str:
        if not value or value.strip() != value or " " in value:
            raise ValueError("record IDs must be non-empty and contain no spaces")
        return value


class DatasetRecord(Record):
    name: str
    version: str
    source: str
    license: str
    redistributable: bool = False
    splits: dict[str, str]
    preprocessing: str
    dvc_path: str | None = None
    hf_revision: str | None = None
    content_sha256: str | None = None


class RunRecord(Record):
    name: str
    command: list[str]
    config_source: str
    resolved_config: dict[str, Any]
    overrides: list[str] = Field(default_factory=list)
    code_commit: str
    dirty: bool
    dirty_patch_sha256: str | None = None
    python_version: str
    platform: str
    seed: int
    launcher: dict[str, Any]
    dataset_ids: list[str]
    mlflow_run_id: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    checkpoint_ids: list[str] = Field(default_factory=list)
    output_dir: str


class CheckpointRecord(Record):
    run_id: str
    alias: str
    format: str
    path: str
    dvc_path: str | None = None
    hf_revision: str | None = None
    sha256: str
    size_bytes: int
    selection_metric: str | None = None
    selection_value: float | None = None


class EvaluationRecord(Record):
    run_id: str
    checkpoint_id: str
    dataset_id: str
    suite: str
    config: dict[str, Any]
    metrics: dict[str, float]
    raw_artifact: str | None = None
    sha256: str | None = None


class EvidenceEntry(Contract):
    label: str
    kind: Literal["figure", "table", "claim"]
    artifact: str | None = None
    sha256: str | None = None
    generator: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    evaluation_ids: list[str] = Field(default_factory=list)
    dataset_ids: list[str] = Field(default_factory=list)
    checkpoint_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class EvidenceDocument(Contract):
    entries: list[EvidenceEntry] = Field(default_factory=list)


class TemplateInfo(Contract):
    advertised_url: str | None = None
    resolved_url: str | None = None
    available: bool = False
    sha256: str | None = None
    source_commit: str | None = None


class VenueRecord(Contract):
    id: str
    name: str
    year: int
    status: Literal["ready", "awaiting_official_template", "manual_review"]
    fetched_at: datetime | None = None
    source_urls: list[str]
    source_hashes: dict[str, str] = Field(default_factory=dict)
    deadlines: dict[str, str] = Field(default_factory=dict)
    timezone: str = "AoE"
    submission_url: str | None = None
    anonymous: bool = True
    page_limit: int | None = None
    template: TemplateInfo
    notes: list[str] = Field(default_factory=list)


class OverleafConfig(Contract):
    project_id: str | None = None
    remote_url: str | None = None
    branch: str = "master"
    head: str | None = None
    authority: Literal["overleaf"] = "overleaf"


class ReleaseManifest(Contract):
    version: str
    status: Literal["candidate", "frozen", "published"]
    generated_at: datetime | None = None
    code_commit: str
    overleaf_head: str | None = None
    datasets: list[str] = Field(default_factory=list)
    runs: list[str] = Field(default_factory=list)
    checkpoints: list[str] = Field(default_factory=list)
    evaluations: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    hf_revisions: dict[str, str] = Field(default_factory=dict)
    checksums: dict[str, str] = Field(default_factory=dict)
    unlinked: dict[str, list[str]] = Field(default_factory=dict)
    supersedes: str | None = None


RECORD_MODELS: dict[str, type[Record]] = {
    "datasets": DatasetRecord,
    "runs": RunRecord,
    "checkpoints": CheckpointRecord,
    "evaluations": EvaluationRecord,
}
