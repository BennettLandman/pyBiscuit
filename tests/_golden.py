"""Locate committed goldens, preferring image goldens over oracle goldens.

``tests/data/golden-image/<tool>/<case>/`` (harvested from ``biscuit_FC_v2.2.sif``, brief 10)
is the acceptance reference; ``tests/data/golden/<tool>/<case>/`` holds the provisional upstream
oracle outputs.  Use :func:`golden_dir` instead of building the path by hand so a test switches
to the image golden as soon as one is committed.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).parent / "data"
ORACLE_GOLDEN = DATA / "golden"
IMAGE_GOLDEN = DATA / "golden-image"


def golden_dir(tool: str, case: str, *, root: Path | None = None) -> Path:
    """Image golden if present, else the oracle golden (which may not exist either)."""
    base = root or DATA
    image = base / "golden-image" / tool / case
    return image if image.is_dir() else base / "golden" / tool / case


def golden_source(tool: str, case: str, *, root: Path | None = None) -> str:
    """``"image"``, ``"oracle"`` or ``"missing"`` for the directory :func:`golden_dir` returns."""
    d = golden_dir(tool, case, root=root)
    if not d.is_dir():
        return "missing"
    meta = d / "command.json"
    oracle = json.loads(meta.read_text()).get("oracle", "") if meta.exists() else ""
    return "image" if oracle.startswith("image-") else "oracle"
