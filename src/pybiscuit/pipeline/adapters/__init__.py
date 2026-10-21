"""Input adapters: locate and verify the products a run starts from."""

from __future__ import annotations

from pathlib import Path

from ...config import Config
from ..model import InputError, InputFindings
from .freesurfer import find_freesurfer, is_freesurfer_subject
from .macruise_v3 import find_macruise_v3, is_macruise_v3
from .pymacruise import find_pymacruise, is_pymacruise_run

__all__ = [
    "ADAPTERS",
    "detect_adapter",
    "find_freesurfer",
    "find_inputs",
    "find_macruise_v3",
    "find_pymacruise",
]

ADAPTERS = ("pymacruise", "macruise-v3", "freesurfer")


def detect_adapter(path: str | Path) -> str:
    if is_pymacruise_run(path):
        return "pymacruise"
    if is_macruise_v3(path):
        return "macruise-v3"
    if is_freesurfer_subject(path):
        return "freesurfer"
    raise InputError(
        f"{path}: cannot tell what this input is. Expected a PyMaCRUISE run (run.json), a "
        "MaCRUISE v3 output directory (Surfaces_FreeView/), or a FreeSurfer subject (surf/). "
        "Use --input-type to force one."
    )


def find_inputs(
    path: str | Path,
    kind: str | None = None,
    config: Config | None = None,
    *,
    allow_incomplete: bool = False,
) -> InputFindings:
    kind = kind or detect_adapter(path)
    if kind == "pymacruise":
        return find_pymacruise(path, allow_incomplete=allow_incomplete)
    if kind == "macruise-v3":
        return find_macruise_v3(path, config)
    if kind == "freesurfer":
        return find_freesurfer(path, config)
    raise InputError(f"unknown input type {kind!r}; choose from {ADAPTERS}")
