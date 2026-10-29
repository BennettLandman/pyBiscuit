# ruff: noqa: E501
"""Locate the open-examples data (ABIDE subjects with real ``biscuit_FC_v2.2`` outputs), never in this repo's history.

Search order: env ``PYBISCUIT_OPEN_EXAMPLES``, ``tests/data/open-examples/``, ``~/dev/biscuit/open-examples``.
Copies may be gzip-compressed (``*.vtk.gz``, ``*.txt.gz``, any ``*.gz``): :func:`subject_dir` returns a directory in
which every file is present uncompressed (symlinks for plain files, decompressed copies for ``.gz`` ones; plain
subject directories are returned as they are).
"""

from __future__ import annotations

import gzip
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = "PYBISCUIT_OPEN_EXAMPLES"


def candidates() -> list[Path]:
    out = []
    if os.environ.get(ENV):
        out.append(Path(os.environ[ENV]).expanduser())
    out += [HERE / "data" / "open-examples", Path.home() / "dev" / "biscuit" / "open-examples"]
    return out


def open_examples_root(root=None) -> Path | None:
    """First existing candidate (or ``root`` if given and existing); ``None`` when there is none."""
    for p in [Path(root).expanduser()] if root else candidates():
        if p.is_dir() and any(c.is_dir() for c in p.iterdir()):  # an unpopulated submodule is empty
            return p
    return None


def _has_gz(d: Path) -> bool:
    return any(d.rglob("*.gz"))


def subject_dir(root: Path, subject: str, cache: Path | None = None) -> Path:
    """Directory of ``subject`` with all files uncompressed (``cache`` receives the materialised copy)."""
    src = Path(root) / subject
    if not _has_gz(src):
        return src
    if cache is None:
        raise ValueError("compressed subject needs a cache directory")
    dst = Path(cache) / subject
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        target = dst / (rel.with_suffix("") if f.suffix == ".gz" else rel)
        if target.exists() or (f.suffix != ".gz" and (src / rel).with_name(rel.name + ".gz").exists()):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix == ".gz":
            with gzip.open(f, "rb") as fi, open(target, "wb") as fo:
                shutil.copyfileobj(fi, fo)
        else:
            target.symlink_to(f.resolve())
    return dst
