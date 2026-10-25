from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest

from researchctl.io import atomic_write_yaml, load_yaml, sha256_bytes
from researchctl.schema import TemplateInfo, VenueRecord
from researchctl.venue import ICLRAdapter, install_template

GUIDELINES = """
<html><body>
<p>ICLR 2027 is double blind.</p>
<p>Abstract submission deadline September 11 2026 AOE.</p>
<p>Full paper submission deadline September 16 2026 11:59pm AOE.</p>
<p>The main text should be 9 pages or fewer.</p>
<a href="https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip">
  ICLR 2027 style files
</a>
</body></html>
"""


def _zip(files: dict[str, bytes]) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return payload.getvalue()


def test_iclr_dead_template_never_falls_back() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    record = ICLRAdapter(2027, client=client, guidelines_html=GUIDELINES).fetch()
    assert record.status == "awaiting_official_template"
    assert record.template.available is False
    assert record.page_limit == 9
    assert record.deadlines["abstract"].startswith("September 11")
    assert "2026" not in (record.template.resolved_url or "")
    assert record.template.advertised_url.endswith("iclr-2027-style-files.zip")


def test_iclr_discovers_relative_official_template_url() -> None:
    html = '<a href="/Conferences/ICLR2027/iclr-2027-style-files.zip">template</a>'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    record = ICLRAdapter(2027, client=client, guidelines_html=html).fetch()
    assert record.template.advertised_url == (
        "https://iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip"
    )


def test_iclr_ignores_untrusted_template_url() -> None:
    html = '<a href="https://example.test/iclr-2027-style-files.zip">template</a>'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    record = ICLRAdapter(2027, client=client, guidelines_html=html).fetch()
    assert record.template.advertised_url.endswith("Master-Template/raw/master/iclr2027.zip")


def test_template_install_requires_review_and_checksum(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _zip({"iclr2027/main.tex": b"official template\n", "iclr2027/style.sty": b"style"})
    checksum = sha256_bytes(payload)
    venue = VenueRecord.model_validate(load_yaml(project / "paper/target/venue.yaml"))
    ready = venue.model_copy(
        update={
            "status": "ready",
            "template": TemplateInfo(
                advertised_url="https://example.test/iclr2027.zip",
                resolved_url="https://example.test/iclr2027.zip",
                available=True,
                sha256=checksum,
            ),
        }
    )
    atomic_write_yaml(project / "paper/target/venue.yaml", ready)

    def fake_get(*args: object, **kwargs: object) -> httpx.Response:
        request = httpx.Request("GET", "https://example.test/iclr2027.zip")
        return httpx.Response(200, content=payload, request=request)

    monkeypatch.setattr("researchctl.venue.httpx.get", fake_get)
    original = (project / "paper/main.tex").read_bytes()
    report = install_template(project, apply=False)
    assert "`modify` `main.tex`" in report.read_text(encoding="utf-8")
    assert (project / "paper/main.tex").read_bytes() == original
    install_template(project, apply=True)
    assert (project / "paper/main.tex").read_bytes() == b"official template\n"
    assert load_yaml(project / "paper/target/template.lock.yaml")["installed"] is True


def test_template_rejects_zip_traversal(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _zip({"../escape.tex": b"bad"})
    checksum = sha256_bytes(payload)
    venue = VenueRecord.model_validate(load_yaml(project / "paper/target/venue.yaml"))
    atomic_write_yaml(
        project / "paper/target/venue.yaml",
        venue.model_copy(
            update={
                "status": "ready",
                "template": TemplateInfo(
                    advertised_url="https://example.test/bad.zip",
                    resolved_url="https://example.test/bad.zip",
                    available=True,
                    sha256=checksum,
                ),
            }
        ),
    )

    def fake_get(*args: object, **kwargs: object) -> httpx.Response:
        request = httpx.Request("GET", "https://example.test/bad.zip")
        return httpx.Response(200, content=payload, request=request)

    monkeypatch.setattr("researchctl.venue.httpx.get", fake_get)
    with pytest.raises(ValueError, match="unsafe path"):
        install_template(project, apply=True)
