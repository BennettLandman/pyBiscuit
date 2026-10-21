"""Label tables for parcellation protocols (``label.csv`` + ``lut.csv``).

``label.csv`` maps names to integer IDs; ``lut.csv`` (``ID,R,G,B``, 0-255) gives the
display colours. When ``lut.csv`` is absent (``check_input`` warns about this) colours
fall back to evenly spaced hues, deterministic in the sorted ID order.
"""

from __future__ import annotations

import colorsys
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LabelTable:
    """``entries``: ID -> (name, (R, G, B)) with RGB in 0-255."""

    entries: dict[int, tuple[str, tuple[int, int, int]]]
    colors_from_fallback: bool = False

    def ids(self) -> list[int]:
        return sorted(self.entries)

    def name(self, label_id: int) -> str:
        return self.entries[label_id][0]

    def rgb(self, label_id: int) -> tuple[int, int, int]:
        return self.entries[label_id][1]


def hsv_fallback_colors(ids) -> dict[int, tuple[int, int, int]]:
    ids = sorted(ids)
    n = max(len(ids), 1)
    out = {}
    for k, i in enumerate(ids):
        r, g, b = colorsys.hsv_to_rgb(k / n, 0.75, 0.95)
        out[i] = (round(r * 255), round(g * 255), round(b * 255))
    return out


def _read_names(path) -> dict[int, str]:
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = [h.strip().lower() for h in rows[0]]
    ic = header.index("id")
    nc = header.index("label") if "label" in header else (1 - ic if ic < 2 else 0)
    return {int(r[ic]): r[nc].strip() for r in rows[1:] if r and r[ic].strip()}


def _read_lut(path) -> dict[int, tuple[int, int, int]]:
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r = {k.strip().upper(): v for k, v in r.items()}
            out[int(r["ID"])] = (int(r["R"]), int(r["G"]), int(r["B"]))
    return out


def read_label_table(label_csv=None, lut_csv=None) -> LabelTable:
    """Load a protocol's label table. ``lut_csv`` defaults to ``lut.csv`` next to ``label_csv``."""
    names: dict[int, str] = {}
    if label_csv is not None:
        label_csv = Path(label_csv)
        names = _read_names(label_csv)
        if lut_csv is None and (label_csv.parent / "lut.csv").exists():
            lut_csv = label_csv.parent / "lut.csv"
    colors = _read_lut(lut_csv) if lut_csv is not None else {}
    ids = sorted(set(names) | set(colors))
    fallback = not colors or bool(set(ids) - set(colors))
    if fallback:
        colors = {**hsv_fallback_colors(ids), **colors}
    entries = {i: (names.get(i, f"label_{i}"), colors[i]) for i in ids}
    return LabelTable(entries, colors_from_fallback=fallback)


def read_protocol_table(parc_dir) -> LabelTable:
    """Convenience: ``parc_dir/label.csv`` + ``parc_dir/lut.csv`` (if present)."""
    parc_dir = Path(parc_dir)
    lut = parc_dir / "lut.csv"
    return read_label_table(parc_dir / "label.csv", lut if lut.exists() else None)
