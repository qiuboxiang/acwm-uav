from __future__ import annotations

from typer.testing import CliRunner

from researchctl.cli import app


def test_cli_exposes_control_plane() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("venue", "data", "run", "eval", "paper", "release", "validate"):
        assert command in result.stdout
