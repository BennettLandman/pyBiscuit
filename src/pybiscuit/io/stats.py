"""BIDS-style TSV stats with a JSON sidecar describing the columns."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_stats_tsv(
    path, columns: dict, descriptions: dict | None = None, units: dict | None = None
):
    """Write ``path`` (``*.tsv``) and ``path`` with ``.json`` suffix; return both paths.

    ``columns``: name -> equal-length sequences. Floats use ``%.9g``; NaN is written as ``n/a``.
    """
    path = Path(path)
    names = list(columns)
    cols = [np.asarray(columns[n]) for n in names]
    n = {len(c) for c in cols}
    if len(n) > 1:
        raise ValueError("all columns must have the same length")

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "n/a" if np.isnan(v) else f"{float(v):.9g}"
        return str(v)

    lines = ["\t".join(names)]
    for i in range(n.pop() if n else 0):
        lines.append("\t".join(fmt(c[i].item() if hasattr(c[i], "item") else c[i]) for c in cols))
    path.write_text("\n".join(lines) + "\n")
    side = {}
    for name in names:
        entry = {"Description": (descriptions or {}).get(name, name)}
        if units and name in units:
            entry["Units"] = units[name]
        side[name] = entry
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(side, indent=2) + "\n")
    return path, sidecar


def read_stats_tsv(path) -> dict[str, list[str]]:
    rows = [r.split("\t") for r in Path(path).read_text().splitlines()]
    return {h: [r[i] for r in rows[1:]] for i, h in enumerate(rows[0])}
