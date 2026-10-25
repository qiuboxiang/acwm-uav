"""Conference adapters and verified template installation."""

from __future__ import annotations

import re
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from researchctl.io import (
    atomic_write_text,
    atomic_write_yaml,
    load_yaml,
    sha256_bytes,
    sha256_file,
)
from researchctl.schema import TemplateInfo, VenueRecord


class VenueAdapter(Protocol):
    def fetch(self) -> VenueRecord: ...


class ICLRAdapter:
    def __init__(
        self,
        year: int,
        client: httpx.Client | None = None,
        guidelines_html: str | None = None,
    ) -> None:
        self.year = year
        self.guidelines_url = f"https://iclr.cc/Conferences/{year}/AuthorGuidelines"
        self.openreview_url = f"https://openreview.net/group?id=ICLR.cc/{year}/Conference"
        self.template_repository = "https://github.com/ICLR/Master-Template"
        self.advertised_template = (
            f"https://github.com/ICLR/Master-Template/raw/master/iclr{year}.zip"
        )
        self.client = client or httpx.Client(
            follow_redirects=True,
            timeout=30,
            headers={"User-Agent": "ai-research-template/0.1 venue-audit"},
        )
        self.guidelines_html = guidelines_html

    @staticmethod
    def _extract_deadline(text: str, keywords: tuple[str, ...]) -> str | None:
        for keyword in keywords:
            pattern = re.compile(
                rf"{keyword}.{{0,100}}?((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                rf"[a-z]*\s+\d{{1,2}}(?:,?\s+\d{{4}})?(?:\s+\d{{1,2}}:\d{{2}}\s*(?:am|pm)?)?"
                rf"(?:\s*(?:AOE|AoE|anywhere on earth))?)",
                re.IGNORECASE | re.DOTALL,
            )
            match = pattern.search(text)
            if match:
                return " ".join(match.group(1).split())
        return None

    def _extract_template_url(self, html: str) -> str | None:
        """Find this year's official ZIP without guessing or crossing conference years."""
        soup = BeautifulSoup(html, "html.parser")
        candidates: list[tuple[int, str]] = []
        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href")
            if not isinstance(href, str):
                continue
            candidate = urljoin(self.guidelines_url, href.strip())
            parsed = urlsplit(candidate)
            hostname = (parsed.hostname or "").lower()
            if parsed.scheme != "https" or not parsed.path.lower().endswith(".zip"):
                continue
            trusted = hostname in {"github.com", "iclr.cc"} or hostname.endswith(".iclr.cc")
            if not trusted:
                continue
            searchable = f"{candidate} {anchor.get_text(' ', strip=True)}".lower()
            year_tokens = (str(self.year), f"iclr{self.year}", f"iclr-{self.year}")
            if not any(token in searchable for token in year_tokens):
                continue
            score = 0
            if hostname == "iclr.cc" or hostname.endswith(".iclr.cc"):
                score += 4
            if any(token in searchable for token in ("template", "style file", "style-file")):
                score += 2
            if f"iclr-{self.year}" in searchable or f"iclr{self.year}" in searchable:
                score += 1
            candidates.append((score, candidate))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[0])[1]

    def fetch(self) -> VenueRecord:
        if self.guidelines_html is None:
            response = self.client.get(self.guidelines_url)
            response.raise_for_status()
            html = response.text
        else:
            html = self.guidelines_html
        text = " ".join(BeautifulSoup(html, "html.parser").get_text(" ").split())
        source_hashes = {self.guidelines_url: sha256_bytes(html.encode())}
        for provenance_url in (self.openreview_url, self.template_repository):
            try:
                provenance_response = self.client.get(provenance_url)
                if provenance_response.status_code == 200:
                    source_hashes[provenance_url] = sha256_bytes(provenance_response.content)
            except httpx.HTTPError:
                continue
        advertised = self._extract_template_url(html) or self.advertised_template
        template_available = False
        resolved_url: str | None = None
        template_hash: str | None = None
        try:
            template_response = self.client.get(advertised)
            if template_response.status_code == 200:
                with tempfile.NamedTemporaryFile(suffix=".zip") as handle:
                    handle.write(template_response.content)
                    handle.flush()
                    template_available = zipfile.is_zipfile(handle.name)
                if template_available:
                    resolved_url = str(template_response.url)
                    template_hash = sha256_bytes(template_response.content)
                    source_hashes[advertised] = template_hash
        except httpx.HTTPError:
            template_available = False
        abstract_deadline = self._extract_deadline(
            text, ("abstract submission deadline", "paper abstracts by")
        )
        paper_deadline = self._extract_deadline(
            text, ("full paper submission deadline", "paper submission deadline")
        )
        deadlines = {}
        if abstract_deadline:
            deadlines["abstract"] = abstract_deadline
        if paper_deadline:
            deadlines["paper"] = paper_deadline
        page_match = re.search(r"main text should be\s+(\d+)\s+pages? or fewer", text, re.I)
        source_commit: str | None = None
        advertised_parts = urlsplit(advertised)
        if template_available and advertised_parts.hostname == "github.com":
            commit_url = (
                "https://api.github.com/repos/ICLR/Master-Template/commits"
                f"?path=iclr{self.year}.zip&per_page=1"
            )
            try:
                commit_response = self.client.get(commit_url)
                if commit_response.status_code == 200:
                    commit_data = commit_response.json()
                    if isinstance(commit_data, list) and commit_data:
                        candidate = commit_data[0]
                        if isinstance(candidate, dict) and isinstance(candidate.get("sha"), str):
                            source_commit = candidate["sha"]
            except (httpx.HTTPError, ValueError):
                source_commit = None
        record = VenueRecord(
            id=f"iclr-{self.year}",
            name="International Conference on Learning Representations",
            year=self.year,
            status="ready" if template_available else "awaiting_official_template",
            fetched_at=datetime.now(UTC),
            source_urls=list(
                dict.fromkeys(
                    [
                        self.guidelines_url,
                        self.openreview_url,
                        self.template_repository,
                        advertised,
                    ]
                )
            ),
            source_hashes=source_hashes,
            deadlines=deadlines,
            timezone="AoE",
            submission_url=self.openreview_url,
            anonymous="double blind" in text.lower() or "double-blind" in text.lower(),
            page_limit=int(page_match.group(1)) if page_match else None,
            template=TemplateInfo(
                advertised_url=advertised,
                resolved_url=resolved_url,
                available=template_available,
                sha256=template_hash,
                source_commit=source_commit,
            ),
            notes=["Never use a prior-year template automatically."],
        )
        return record


def adapter_for(target: str) -> VenueAdapter:
    match = re.fullmatch(r"iclr-(\d{4})", target.lower())
    if match:
        return ICLRAdapter(int(match.group(1)))
    raise ValueError(f"no live adapter for {target}; provide a reviewed paper/target/venue.yaml")


def sync_venue(root: Path, target: str) -> VenueRecord:
    record = adapter_for(target).fetch()
    path = root / "paper/target/venue.yaml"
    if path.is_file():
        existing = VenueRecord.model_validate(load_yaml(path))
        ignored = {"fetched_at", "source_hashes"}
        current_data = record.model_dump(mode="json", exclude=ignored)
        existing_data = existing.model_dump(mode="json", exclude=ignored)
        hashes_complete = set(record.source_hashes).issubset(existing.source_hashes)
        if current_data == existing_data and hashes_complete:
            return existing
    atomic_write_yaml(path, record)
    return record


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    safe: list[zipfile.ZipInfo] = []
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe path in template archive: {member.filename}")
        if member.is_dir():
            continue
        safe.append(member)
    if not safe:
        raise ValueError("template archive contains no files")
    return safe


def _strip_common_root(names: list[PurePosixPath]) -> list[PurePosixPath]:
    if names and all(len(path.parts) > 1 for path in names):
        root_name = names[0].parts[0]
        if all(path.parts[0] == root_name for path in names):
            return [PurePosixPath(*path.parts[1:]) for path in names]
    return names


def install_template(
    root: Path,
    *,
    source_url: str | None = None,
    expected_sha256: str | None = None,
    apply: bool = False,
) -> Path:
    venue = VenueRecord.model_validate(load_yaml(root / "paper/target/venue.yaml"))
    url = source_url or venue.template.resolved_url
    checksum = expected_sha256 or venue.template.sha256
    if source_url and not expected_sha256:
        raise ValueError("a manual template override requires --sha256")
    if not url or not checksum:
        raise ValueError("official template is not yet verified; provide --source-url and --sha256")
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    actual = sha256_bytes(response.content)
    if actual != checksum:
        raise ValueError(f"template checksum mismatch: expected {checksum}, got {actual}")
    paper_root = root / "paper"
    changes: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory() as temporary_name:
        temporary = Path(temporary_name)
        archive_path = temporary / "template.zip"
        archive_path.write_bytes(response.content)
        with zipfile.ZipFile(archive_path) as archive:
            members = _safe_members(archive)
            names = _strip_common_root([PurePosixPath(item.filename) for item in members])
            for member, normalized in zip(members, names, strict=True):
                destination = paper_root / normalized.as_posix()
                content = archive.read(member)
                state = "add"
                if destination.exists():
                    state = (
                        "unchanged"
                        if sha256_bytes(content) == sha256_file(destination)
                        else "modify"
                    )
                changes.append((state, normalized.as_posix()))
                if apply and state != "unchanged":
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
    report = root / "paper/target/template-diff.md"
    atomic_write_text(
        report,
        "<!-- Generated by researchctl venue install-template. -->\n\n"
        "# Template change report\n\n"
        + "".join(f"- `{state}` `{path}`\n" for state, path in changes),
    )
    if not apply:
        return report
    atomic_write_yaml(
        root / "paper/target/template.lock.yaml",
        {
            "schema_version": 1,
            "installed": True,
            "source_url": url,
            "sha256": actual,
            "source_commit": venue.template.source_commit,
            "installed_at": datetime.now(UTC).isoformat(),
        },
    )
    return report
