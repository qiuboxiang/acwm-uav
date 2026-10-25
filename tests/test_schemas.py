from __future__ import annotations

import json
from pathlib import Path

from researchctl.schema_export import SCHEMA_MODELS, export_schemas, render_schemas


def test_exported_schemas_are_current_and_valid_json(project: Path) -> None:
    paths = export_schemas(project)
    assert len(paths) == len(SCHEMA_MODELS)
    rendered = render_schemas()
    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert content == rendered[path.name]
        schema = json.loads(content)
        assert schema["type"] == "object"
