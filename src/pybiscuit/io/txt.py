"""Per-vertex text files: one value per line (e.g. ``lh.curv.mean.txt``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_vertex_txt(path, dtype=np.float64, n_vertices: int | None = None) -> np.ndarray:
    """Read a whitespace/newline separated list of numbers into a 1-D array.

    If ``n_vertices`` is given the count must match, otherwise ``ValueError``.
    """
    text = Path(path).read_text()
    if not text.strip():
        out = np.empty(0, dtype=dtype)
    else:
        out = np.fromstring(text, sep=" ", dtype=np.float64)
        if out.size != len(text.split()):
            raise ValueError(f"{path}: could not parse all values as numbers")
        out = out.astype(dtype, copy=False)
    if n_vertices is not None and out.shape[0] != n_vertices:
        raise ValueError(f"{path}: expected {n_vertices} values, found {out.shape[0]}")
    return out


def write_vertex_txt(values, path, fmt: str | None = None) -> None:
    """Write a 1-D array, one value per line. Default format round-trips exactly."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("per-vertex values must be 1-D")
    if fmt is None:
        fmt = "%d" if arr.dtype.kind in "iub" else ("%.9g" if arr.dtype == np.float32 else "%.17g")
    with open(path, "w") as fh:
        if arr.size:
            fh.write("\n".join(fmt % v for v in arr.tolist()))
            fh.write("\n")
