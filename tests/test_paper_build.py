from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from researchctl.paper import build_paper


@pytest.mark.integration
def test_reference_paper_builds(project: Path) -> None:
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk is not installed")
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is not None:
        dependency = subprocess.run(
            [kpsewhich, "eso-pic.sty"], capture_output=True, check=False, text=True
        )
        if dependency.returncode != 0:
            pytest.skip("local TeX Live lacks the ICLR template dependency eso-pic.sty")
    assert build_paper(project).is_file()
