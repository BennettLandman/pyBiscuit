"""Colour conventions: fixed limits per measure, matching the original BISCUIT PDF.

Limits (docs/visualization-plan.md, page 5): mean curvature +-0.05, shape index +-1,
thickness 1-5 mm, sulcal depth 0-30 mm, LGI 2-10, SCI 0-1. Magnitudes use ``viridis``,
signed measures use the diverging ``RdBu_r``. The template feature means (``curv``, ``sulc``,
``iH``) are not in the PDF; their limits here are a display convention of this package.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeasureStyle:
    name: str
    vmin: float
    vmax: float
    cmap: str
    label: str
    source: str = "pdf"  # "pdf" = from the original report; "pybiscuit" = our convention

    @property
    def limits(self) -> tuple[float, float]:
        return (self.vmin, self.vmax)


def _sym(name, lim, label, source="pdf"):
    return MeasureStyle(name, -lim, lim, "RdBu_r", label, source)


MEASURES: dict[str, MeasureStyle] = {
    s.name: s
    for s in (
        _sym("mean_curvature", 0.05, "mean curvature (1/mm)"),
        _sym("shape_index", 1.0, "shape index"),
        MeasureStyle("thickness", 1.0, 5.0, "viridis", "thickness (mm)"),
        MeasureStyle("sulcal_depth", 0.0, 30.0, "viridis", "sulcal depth (mm)"),
        MeasureStyle("lgi", 2.0, 10.0, "viridis", "LGI"),
        MeasureStyle("sci", 0.0, 1.0, "viridis", "SCI"),
        _sym("curv", 0.5, "curvature (template)", "pybiscuit"),
        _sym("sulc", 1.5, "sulc (template)", "pybiscuit"),
        _sym("iH", 2.5, "integrated mean curvature (template)", "pybiscuit"),
    )
}

_ALIASES = {
    "h": "mean_curvature",
    "mean_curv": "mean_curvature",
    "si": "shape_index",
    "shapeindex": "shape_index",
    "depth": "sulcal_depth",
    "sulcaldepth": "sulcal_depth",
    "thick": "thickness",
    "gyrification": "lgi",
}


def measure_style(name: str, data=None) -> MeasureStyle:
    """Style for a measure name (case-insensitive; aliases accepted).

    Unknown measures get robust 2-98 percentile limits from ``data`` (viridis), or raise
    ``KeyError`` when no data is supplied.
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    for k in (name, key):
        if k in MEASURES:
            return MEASURES[k]
    if key in _ALIASES:
        return MEASURES[_ALIASES[key]]
    # strip hemisphere prefix / extensions like "lh.curv" or "rh_lgi"
    for sep in (".", "_"):
        head, _, tail = key.partition(sep)
        if head in ("lh", "rh") and tail:
            return measure_style(tail, data)
    if data is None:
        raise KeyError(name)
    import numpy as np

    lo, hi = np.nanpercentile(np.asarray(data, float), [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    return MeasureStyle(name, float(lo), float(hi), "viridis", name, "data")
