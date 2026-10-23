"""BISCUIT Phase 4 summary statistics and registered-space property maps (`B10.1`, spec §10).

Port of ``BISCUIT-master/SurfQuantStat/surf_quant_stat_pipeline.m`` (lines 1-302, 391-464),
``BISCUIT-master/SurfQuantStat/IO/read_bary.m`` (lines 1-15), and the Phase 4 file staging in
``BISCUIT-master/BISCUIT/GenericRecon/biscuit.m`` (lines 411-470).
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..sphere.remesh import resample

__all__ = [
    "AGGREGATE_CORTEX_ID",
    "AGGREGATE_HEMI_IDS",
    "AGGREGATE_LABELS",
    "AGGREGATE_LOBE_IDS",
    "CSV_COLUMNS",
    "MEASURES",
    "ProtocolRow",
    "StatsTable",
    "assign_curve_parcels",
    "bary_interpolate",
    "compute_roi_and_curve_stats",
    "format_stat_float",
    "read_bary",
    "read_protocol_labels",
    "read_stats_csv",
    "remesh_stat_maps",
    "run_stats_pipeline",
    "write_stats_csv",
]

#: Measure order in ``surf_quant_stat_pipeline.m:395-406, 229-254``
MEASURES: tuple[str, ...] = ("H", "S", "Depth", "Thickness", "SCI", "LGI")

#: Column names in ``stats.<protocol>.csv`` (``surf_quant_stat_pipeline.m:13-14, 228-254``)
CSV_COLUMNS: tuple[str, ...] = (
    "ID",
    "Label",
    "nVertices",
    "MeanCurvature",
    "MeanCurvature_stdev",
    "ShapeIndex",
    "ShapeIndex_stdev",
    "SulcalDepth",
    "SulcalDepth_stdev",
    "Thickness",
    "Thickness_stdev",
    "SCI",
    "SCI_stdev",
    "LGI",
    "LGI_stdev",
    "nVertices_curve",
    "MeanCurvature_curve",
    "MeanCurvature_stdev_curve",
    "ShapeIndex_curve",
    "ShapeIndex_stdev_curve",
    "SulcalDepth_curve",
    "SulcalDepth_stdev_curve",
    "Thickness_curve",
    "Thickness_stdev_curve",
    "SCI_curve",
    "SCI_stdev_curve",
    "LGI_curve",
    "LGI_stdev_curve",
)

AGGREGATE_CORTEX_ID = 20000
AGGREGATE_HEMI_IDS: tuple[int, int] = (20001, 20002)
AGGREGATE_LOBE_IDS: tuple[int, ...] = tuple(range(20003, 20013))

AGGREGATE_LABELS: dict[int, str] = {
    20000: "Whole cortex",
    20001: "Left hemisphere",
    20002: "Right hemisphere",
    20003: "Left frontal lobe",
    20004: "Right frontal lobe",
    20005: "Left temporal lobe",
    20006: "Right temporal lobe",
    20007: "Left parietal lobe",
    20008: "Right parietal lobe",
    20009: "Left occipital lobe",
    20010: "Right occipital lobe",
    20011: "Left insula",
    20012: "Right insula",
}

_MEASURE_ALIASES: dict[str, tuple[str, ...]] = {
    "H": ("H", "MeanCurvature", "mid.H"),
    "S": ("S", "ShapeIndex", "mid.S"),
    "Depth": ("Depth", "SD", "SulcalDepth", "gray.SD", "mid.SD"),
    "Thickness": ("Thickness", "CT", "thickness", "mid.CT"),
    "SCI": ("SCI", "mid.SCI"),
    "LGI": ("LGI", "Gyrification", "gray.LGI", "mid.LGI"),
}


@dataclass(frozen=True)
class ProtocolRow:
    """One entry in ``labelmap`` (with prepended ``Unknown, 0`` as row 0)."""

    label: str
    id: int
    lobe: int | None = None


@dataclass
class StatsTable:
    """Computed ROI, curve, and aggregate statistics for one parcellation protocol."""

    ids: np.ndarray  # int64 (R,)
    labels: list[str]  # length R
    n_vertices: np.ndarray  # int64 (R,)
    avg: np.ndarray  # float64 (R, 6) in MEASURES order
    stdev: np.ndarray  # float64 (R, 6) in MEASURES order
    n_vertices_curve: np.ndarray  # int64 (R,)
    avg_curve: np.ndarray  # float64 (R, 6) in MEASURES order
    stdev_curve: np.ndarray  # float64 (R, 6) in MEASURES order

    def __len__(self) -> int:
        return len(self.ids)

    def row(self, region_id: int) -> dict[str, Any]:
        """Return the 28-column dict for ``region_id`` (raises ``KeyError`` if absent)."""
        matches = np.flatnonzero(self.ids == int(region_id))
        if matches.size == 0:
            raise KeyError(f"region ID {region_id} not in StatsTable")
        return self._row_dict(int(matches[0]))

    def _row_dict(self, i: int) -> dict[str, Any]:
        d: dict[str, Any] = {
            "ID": int(self.ids[i]),
            "Label": self.labels[i],
            "nVertices": int(self.n_vertices[i]),
        }
        names = (
            "MeanCurvature",
            "ShapeIndex",
            "SulcalDepth",
            "Thickness",
            "SCI",
            "LGI",
        )
        for m, name in enumerate(names):
            d[name] = float(self.avg[i, m])
            d[f"{name}_stdev"] = float(self.stdev[i, m])
        d["nVertices_curve"] = int(self.n_vertices_curve[i])
        for m, name in enumerate(names):
            d[f"{name}_curve"] = float(self.avg_curve[i, m])
            d[f"{name}_stdev_curve"] = float(self.stdev_curve[i, m])
        return d

    def to_rows(self) -> list[dict[str, Any]]:
        """Return a list of dicts keyed by :data:`CSV_COLUMNS`."""
        return [self._row_dict(i) for i in range(len(self.ids))]

    def to_columns(self) -> dict[str, list[Any]]:
        """Return ``{col_name: list_of_values}`` in :data:`CSV_COLUMNS` order."""
        rows = self.to_rows()
        return {col: [r[col] for r in rows] for col in CSV_COLUMNS}

    def write_csv(self, path: str | Path) -> Path:
        """Write ``stats.<protocol>.csv`` and return ``Path(path)``."""
        return write_stats_csv(path, self)


def read_bary(
    path: str | Path, *, unique: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a TRACE ``.bary`` file (``read_bary.m`` + ``unique(curve, 'rows')``).

    Format:
    - Line 1: number of curves ``C``.
    - Per curve: a line with the point count ``P`` (skipping blank lines), followed by ``P``
      rows of ``v1 v2 w`` (``%d %d %f``, 0-based vertex indices and barycentric weight on ``v1``).

    When ``unique=True`` (the default, matching ``surf_quant_stat_pipeline.m:37-38``), all rows
    across all curves are stacked and deduplicated in lexicographic row order ``(v1, v2, w)``,
    removing duplicate junction points shared by multiple curves.

    Returns ``(v1, v2, w)`` where ``v1, v2`` are 0-based ``int64`` 1-D arrays and ``w`` is a
    ``float64`` 1-D array.
    """
    text = Path(path).read_text()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    n_curves = int(float(lines[0]))
    idx = 1
    rows: list[tuple[int, int, float]] = []
    for _ in range(n_curves):
        if idx >= len(lines):
            raise ValueError(f"{path}: unexpected EOF reading curve point count")
        n_pts = int(float(lines[idx]))
        idx += 1
        if idx + n_pts > len(lines):
            raise ValueError(f"{path}: unexpected EOF reading {n_pts} barycentric points")
        for j in range(idx, idx + n_pts):
            parts = lines[j].split()
            if len(parts) < 3:
                raise ValueError(f"{path}: malformed barycentric row {lines[j]!r}")
            rows.append((int(parts[0]), int(parts[1]), float(parts[2])))
        idx += n_pts
    if not rows:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    arr = np.array(rows, dtype=np.float64)
    if unique:
        arr = np.unique(arr, axis=0)
    return (
        arr[:, 0].astype(np.int64),
        arr[:, 1].astype(np.int64),
        arr[:, 2].astype(np.float64),
    )


def _coerce_bary(
    curve: str | Path | tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(curve, (str, Path)):
        return read_bary(curve, unique=True)
    if isinstance(curve, tuple) and len(curve) == 3:
        v1, v2, w = curve
        return (
            np.asarray(v1, dtype=np.int64),
            np.asarray(v2, dtype=np.int64),
            np.asarray(w, dtype=np.float64),
        )
    arr = np.asarray(curve, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[1] == 3:
        if len(arr):
            arr = np.unique(arr, axis=0)
        return (
            arr[:, 0].astype(np.int64),
            arr[:, 1].astype(np.int64),
            arr[:, 2].astype(np.float64),
        )
    raise TypeError("curve must be a path, a (v1, v2, w) 3-tuple, or an (N, 3) array")


def _coerce_props(props: np.ndarray | Mapping[str, np.ndarray]) -> np.ndarray:
    if isinstance(props, Mapping):
        cols = []
        for m in MEASURES:
            found = None
            for alias in _MEASURE_ALIASES[m]:
                if alias in props:
                    found = np.asarray(props[alias], dtype=np.float64)
                    break
            if found is None:
                raise KeyError(f"missing measure {m!r} (tried aliases {_MEASURE_ALIASES[m]})")
            cols.append(found.reshape(-1))
        return np.column_stack(cols)
    arr = np.asarray(props, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(MEASURES):
        raise ValueError(f"property matrix must have shape (N, 6), got {arr.shape}")
    return arr


def read_protocol_labels(
    label_csv: str | Path | Sequence[ProtocolRow | tuple[Any, ...]],
) -> tuple[list[ProtocolRow], bool]:
    """Read protocol ``label.csv`` (header line skipped) and prepend ``Unknown, 0``.

    Matches ``surf_quant_stat_pipeline.m:11-27``:
    - Column 0: ``Label``
    - Column 1: ``ID``
    - Column 2 (optional, e.g. ``braincolor``): ``lobe`` in ``1..5``
    - Prepends ``ProtocolRow("Unknown", 0, 0 if has_lobe else None)`` as entry 0.

    Returns ``(rows, has_lobe)``.
    """
    raw_rows: list[tuple[str, int, int | None]] = []
    if isinstance(label_csv, (str, Path)):
        with open(label_csv, newline="") as fh:
            reader = list(csv.reader(fh))
        data_rows = [r for r in reader[1:] if r and any(c.strip() for c in r)]
        has_lobe = any(len(r) >= 3 and r[2].strip() != "" for r in data_rows)
        for r in data_rows:
            name = r[0].strip()
            rid = int(r[1].strip())
            lobe = int(r[2].strip()) if has_lobe else None
            raw_rows.append((name, rid, lobe))
    else:
        items = list(label_csv)
        has_lobe = any(
            (
                r.lobe is not None
                if isinstance(r, ProtocolRow)
                else (len(r) >= 3 and r[2] is not None)
            )
            for r in items
        )
        for r in items:
            if isinstance(r, ProtocolRow):
                lobe_val = int(r.lobe) if has_lobe and r.lobe is not None else None
                raw_rows.append((r.label, int(r.id), lobe_val))
            else:
                name = str(r[0]).strip()
                rid = int(r[1])
                lobe = int(r[2]) if has_lobe and len(r) >= 3 and r[2] is not None else None
                raw_rows.append((name, rid, lobe))

    out = [ProtocolRow("Unknown", 0, 0 if has_lobe else None)]
    for name, rid, lobe in raw_rows:
        out.append(ProtocolRow(name, rid, lobe))
    return out, has_lobe


def bary_interpolate(
    pmap: np.ndarray, v1: np.ndarray, v2: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """``bary(pmap, edges)`` in ``surf_quant_stat_pipeline.m:457-459``.

    ``pmap`` may be 1-D ``(N,)`` or 2-D ``(N, M)``. Returns ``pmap[v1] * w + pmap[v2] * (1 - w)``.
    """
    p = np.asarray(pmap, dtype=np.float64)
    v1 = np.asarray(v1, dtype=np.int64)
    v2 = np.asarray(v2, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    if p.ndim == 1:
        if len(v1) == 0:
            return np.empty(0, dtype=np.float64)
        return p[v1] * w + p[v2] * (1.0 - w)
    if len(v1) == 0:
        return np.empty((0, p.shape[1]), dtype=np.float64)
    return p[v1] * w[:, None] + p[v2] * (1.0 - w)[:, None]


def assign_curve_parcels(
    parc: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    w: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Assign each curve point to ``parc[v1]`` if ``w >= threshold`` else ``parc[v2]``."""
    parc = np.asarray(parc, dtype=np.int64)
    v1 = np.asarray(v1, dtype=np.int64)
    v2 = np.asarray(v2, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    if len(v1) == 0:
        return np.empty(0, dtype=np.int64)
    return np.where(w >= threshold, parc[v1], parc[v2])


def _stats_for_regions(
    regions: Sequence[int] | np.ndarray,
    prop_lh: np.ndarray,
    prop_rh: np.ndarray,
    parc_lh: np.ndarray,
    parc_rh: np.ndarray,
    prop_curve_lh: np.ndarray,
    prop_curve_rh: np.ndarray,
    parc_curve_lh: np.ndarray,
    parc_curve_rh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-region mean, sample std (``ddof=1``), and counts across stacked hemispheres."""
    reg = np.asarray(regions, dtype=np.int64)
    n_reg = len(reg)
    n_meas = prop_lh.shape[1]
    avg = np.full((n_reg, n_meas), np.nan, dtype=np.float64)
    stdev = np.full((n_reg, n_meas), np.nan, dtype=np.float64)
    nv = np.zeros(n_reg, dtype=np.int64)
    avg_c = np.full((n_reg, n_meas), np.nan, dtype=np.float64)
    stdev_c = np.full((n_reg, n_meas), np.nan, dtype=np.float64)
    nv_c = np.zeros(n_reg, dtype=np.int64)

    for i, r in enumerate(reg):
        data = np.vstack([prop_lh[parc_lh == r], prop_rh[parc_rh == r]])
        n = len(data)
        nv[i] = n
        if n >= 1:
            avg[i] = data.mean(axis=0)
        if n >= 2:
            stdev[i] = data.std(axis=0, ddof=1)

        data_c = np.vstack([prop_curve_lh[parc_curve_lh == r], prop_curve_rh[parc_curve_rh == r]])
        nc = len(data_c)
        nv_c[i] = nc
        if nc >= 1:
            avg_c[i] = data_c.mean(axis=0)
        if nc >= 2:
            stdev_c[i] = data_c.std(axis=0, ddof=1)

    return avg, stdev, nv, avg_c, stdev_c, nv_c


def _ismember_index(values: np.ndarray, region: np.ndarray) -> np.ndarray:
    """0-based index into ``region`` of first match for each element of ``values``."""
    lookup: dict[int, int] = {}
    for idx, r in enumerate(region.tolist()):
        if int(r) not in lookup:
            lookup[int(r)] = idx
    return np.array([lookup.get(int(v), 0) for v in values.tolist()], dtype=np.int64)


def compute_roi_and_curve_stats(
    props_lh: np.ndarray | Mapping[str, np.ndarray],
    props_rh: np.ndarray | Mapping[str, np.ndarray],
    parc_lh: np.ndarray,
    parc_rh: np.ndarray,
    curve_lh: str | Path | tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray,
    curve_rh: str | Path | tuple[np.ndarray, np.ndarray, np.ndarray] | np.ndarray,
    label_csv: str | Path | Sequence[ProtocolRow | tuple[Any, ...]],
    *,
    curve_threshold: float = 0.5,
) -> StatsTable:
    """Compute ROI-wise, curve-wise, and aggregate statistics as in ``surf_quant_stat_pipeline.m``.

    - Vertex labels not present in ``label_csv`` are mapped to ``0`` (``Unknown``).
    - No cortical mask is applied (``surf_quant_stat_pipeline.m`` does not mask by ``mask.txt``).
    - Curve points are deduplicated via ``np.unique(..., axis=0)`` on ``(v1, v2, w)`` and assigned
      to ``parc[v1]`` when ``w >= curve_threshold`` (``0.5``) else ``parc[v2]``.
    - Aggregates:
      - ``20000`` (``Whole cortex``): all vertices / curve points in ``lh`` + ``rh`` whose label
        is in ``region`` after removing ``Unknown``.
      - ``20001`` (``Left hemisphere``), ``20002`` (``Right hemisphere``): vertices / curve points
        in ``lh`` / ``rh`` with a non-zero valid protocol label.
      - ``20003..20012`` (five lobes x ``{Left, Right}``): included iff ``label.csv`` has a third
        lobe column (``1..5`` -> ``(lobe - 1) * 2 + (1 if rh else 0) + 20003``).
    - Finally, any row with ``ID == 0`` (``Unknown``) is removed before returning.
    """
    prop_lh = _coerce_props(props_lh)
    prop_rh = _coerce_props(props_rh)
    p_lh = np.asarray(parc_lh, dtype=np.int64).reshape(-1)
    p_rh = np.asarray(parc_rh, dtype=np.int64).reshape(-1)
    if len(p_lh) != len(prop_lh):
        raise ValueError(f"parc_lh length {len(p_lh)} != prop_lh length {len(prop_lh)}")
    if len(p_rh) != len(prop_rh):
        raise ValueError(f"parc_rh length {len(p_rh)} != prop_rh length {len(prop_rh)}")

    v1_lh, v2_lh, w_lh = _coerce_bary(curve_lh)
    v1_rh, v2_rh, w_rh = _coerce_bary(curve_rh)

    labelmap, has_lobe = read_protocol_labels(label_csv)
    region = np.array([row.id for row in labelmap], dtype=np.int64)

    # parc_lh(~ismember(parc_lh, region)) = 0
    p_lh = np.where(np.isin(p_lh, region), p_lh, 0)
    p_rh = np.where(np.isin(p_rh, region), p_rh, 0)

    # curve parcels & interpolated values
    pc_lh = assign_curve_parcels(p_lh, v1_lh, v2_lh, w_lh, threshold=curve_threshold)
    pc_rh = assign_curve_parcels(p_rh, v1_rh, v2_rh, w_rh, threshold=curve_threshold)
    prop_curve_lh = bary_interpolate(prop_lh, v1_lh, v2_lh, w_lh)
    prop_curve_rh = bary_interpolate(prop_rh, v1_rh, v2_rh, w_rh)

    # 1. Per-ROI stats (including row 0 = Unknown)
    avg, stdev, nv, avg_c, stdev_c, nv_c = _stats_for_regions(
        region, prop_lh, prop_rh, p_lh, p_rh, prop_curve_lh, prop_curve_rh, pc_lh, pc_rh
    )

    # 2. Region combination indices (0-based index into labelmap; > 0 means non-Unknown)
    id_lh = _ismember_index(p_lh, region)
    id_rh = _ismember_index(p_rh, region)
    id_curve_lh = _ismember_index(pc_lh, region)
    id_curve_rh = _ismember_index(pc_rh, region)

    # Left / Right hemispheres (20001, 20002)
    parc_2_lh = np.where(id_lh > 0, 20001, 20000)
    parc_2_rh = np.where(id_rh > 0, 20002, 20000)
    parc_curve_2_lh = np.where(id_curve_lh > 0, 20001, 20000)
    parc_curve_2_rh = np.where(id_curve_rh > 0, 20002, 20000)
    avg_2, stdev_2, nv_2, avg_c_2, stdev_c_2, nv_c_2 = _stats_for_regions(
        AGGREGATE_HEMI_IDS,
        prop_lh,
        prop_rh,
        parc_2_lh,
        parc_2_rh,
        prop_curve_lh,
        prop_curve_rh,
        parc_curve_2_lh,
        parc_curve_2_rh,
    )

    # 5 lobes (20003..20012) when label.csv has >= 3 columns
    if has_lobe:
        lobe_arr = np.array([int(row.lobe or 0) for row in labelmap], dtype=np.int64)
        lobe_lh = lobe_arr[id_lh]
        lobe_rh = lobe_arr[id_rh]
        lobe_c_lh = lobe_arr[id_curve_lh]
        lobe_c_rh = lobe_arr[id_curve_rh]
        parc_5_lh = (lobe_lh - 1) * 2 + (parc_2_lh == 20002).astype(np.int64) + 20003
        parc_5_rh = (lobe_rh - 1) * 2 + (parc_2_rh == 20002).astype(np.int64) + 20003
        parc_c_5_lh = (lobe_c_lh - 1) * 2 + (parc_curve_2_lh == 20002).astype(np.int64) + 20003
        parc_c_5_rh = (lobe_c_rh - 1) * 2 + (parc_curve_2_rh == 20002).astype(np.int64) + 20003
        avg_5, stdev_5, nv_5, avg_c_5, stdev_c_5, nv_c_5 = _stats_for_regions(
            AGGREGATE_LOBE_IDS,
            prop_lh,
            prop_rh,
            parc_5_lh,
            parc_5_rh,
            prop_curve_lh,
            prop_curve_rh,
            parc_c_5_lh,
            parc_c_5_rh,
        )

    # Whole cortex (20000): remove 'Unknown' (case-insensitive) from region first (:120-123)
    valid_regions = np.array(
        [row.id for row in labelmap if row.label.lower() != "unknown"], dtype=np.int64
    )
    mask_1_lh = np.isin(p_lh, valid_regions)
    mask_1_rh = np.isin(p_rh, valid_regions)
    data_1 = np.vstack([prop_lh[mask_1_lh], prop_rh[mask_1_rh]])
    nv_1 = np.array([len(data_1)], dtype=np.int64)
    avg_1 = np.full((1, len(MEASURES)), np.nan, dtype=np.float64)
    stdev_1 = np.full((1, len(MEASURES)), np.nan, dtype=np.float64)
    if len(data_1) >= 1:
        avg_1[0] = data_1.mean(axis=0)
    if len(data_1) >= 2:
        stdev_1[0] = data_1.std(axis=0, ddof=1)

    mask_c1_lh = np.isin(pc_lh, valid_regions)
    mask_c1_rh = np.isin(pc_rh, valid_regions)
    data_c1 = np.vstack([prop_curve_lh[mask_c1_lh], prop_curve_rh[mask_c1_rh]])
    nv_c_1 = np.array([len(data_c1)], dtype=np.int64)
    avg_c_1 = np.full((1, len(MEASURES)), np.nan, dtype=np.float64)
    stdev_c_1 = np.full((1, len(MEASURES)), np.nan, dtype=np.float64)
    if len(data_c1) >= 1:
        avg_c_1[0] = data_c1.mean(axis=0)
    if len(data_c1) >= 2:
        stdev_c_1[0] = data_c1.std(axis=0, ddof=1)

    # Stack rows in exact MATLAB order: labelmap, 20000, 20001..20002, [20003..20012]
    all_ids = [row.id for row in labelmap] + [AGGREGATE_CORTEX_ID, *AGGREGATE_HEMI_IDS]
    all_labels = [row.label for row in labelmap] + [
        AGGREGATE_LABELS[AGGREGATE_CORTEX_ID],
        *(AGGREGATE_LABELS[i] for i in AGGREGATE_HEMI_IDS),
    ]
    all_avg = np.vstack([avg, avg_1, avg_2])
    all_stdev = np.vstack([stdev, stdev_1, stdev_2])
    all_nv = np.concatenate([nv, nv_1, nv_2])
    all_avg_c = np.vstack([avg_c, avg_c_1, avg_c_2])
    all_stdev_c = np.vstack([stdev_c, stdev_c_1, stdev_c_2])
    all_nv_c = np.concatenate([nv_c, nv_c_1, nv_c_2])

    if has_lobe:
        all_ids.extend(AGGREGATE_LOBE_IDS)
        all_labels.extend(AGGREGATE_LABELS[i] for i in AGGREGATE_LOBE_IDS)
        all_avg = np.vstack([all_avg, avg_5])
        all_stdev = np.vstack([all_stdev, stdev_5])
        all_nv = np.concatenate([all_nv, nv_5])
        all_avg_c = np.vstack([all_avg_c, avg_c_5])
        all_stdev_c = np.vstack([all_stdev_c, stdev_c_5])
        all_nv_c = np.concatenate([all_nv_c, nv_c_5])

    # Drop ID == 0 (surf_quant_stat_pipeline.m:257, 260)
    ids_arr = np.array(all_ids, dtype=np.int64)
    keep = ids_arr != 0
    return StatsTable(
        ids=ids_arr[keep],
        labels=[lbl for lbl, k in zip(all_labels, keep.tolist(), strict=True) if k],
        n_vertices=all_nv[keep],
        avg=all_avg[keep],
        stdev=all_stdev[keep],
        n_vertices_curve=all_nv_c[keep],
        avg_curve=all_avg_c[keep],
        stdev_curve=all_stdev_c[keep],
    )


def format_stat_float(v: float) -> str:
    """Format a float cell as R2012a ``export(..., 'Delimiter', ',')``: ``%.15g`` or ``NaN``."""
    if np.isnan(v):
        return "NaN"
    if v == 0.0:
        return "0"
    return f"{float(v):.15g}"


def write_stats_csv(path: str | Path, table: StatsTable) -> Path:
    """Write ``stats.<protocol>.csv`` with the exact 28 columns and ``%.15g`` float formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for i in range(len(table.ids)):
            row: list[str] = [
                str(int(table.ids[i])),
                table.labels[i],
                str(int(table.n_vertices[i])),
            ]
            for m in range(len(MEASURES)):
                row.append(format_stat_float(table.avg[i, m]))
                row.append(format_stat_float(table.stdev[i, m]))
            row.append(str(int(table.n_vertices_curve[i])))
            for m in range(len(MEASURES)):
                row.append(format_stat_float(table.avg_curve[i, m]))
                row.append(format_stat_float(table.stdev_curve[i, m]))
            writer.writerow(row)
    return path


def read_stats_csv(path: str | Path) -> StatsTable:
    """Parse a ``stats.<protocol>.csv`` file back into a :class:`StatsTable`."""
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    header = tuple(rows[0])
    if header != CSV_COLUMNS:
        raise ValueError(f"{path}: unexpected header {header!r}")
    data = rows[1:]
    n = len(data)
    ids = np.empty(n, dtype=np.int64)
    labels: list[str] = []
    nv = np.empty(n, dtype=np.int64)
    avg = np.empty((n, 6), dtype=np.float64)
    stdev = np.empty((n, 6), dtype=np.float64)
    nv_c = np.empty(n, dtype=np.int64)
    avg_c = np.empty((n, 6), dtype=np.float64)
    stdev_c = np.empty((n, 6), dtype=np.float64)
    for i, r in enumerate(data):
        ids[i] = int(r[0])
        labels.append(r[1])
        nv[i] = int(r[2])
        for m in range(6):
            avg[i, m] = float(r[3 + 2 * m])
            stdev[i, m] = float(r[4 + 2 * m])
        nv_c[i] = int(r[15])
        for m in range(6):
            avg_c[i, m] = float(r[16 + 2 * m])
            stdev_c[i, m] = float(r[17 + 2 * m])
    return StatsTable(
        ids=ids,
        labels=labels,
        n_vertices=nv,
        avg=avg,
        stdev=stdev,
        n_vertices_curve=nv_c,
        avg_curve=avg_c,
        stdev_curve=stdev_c,
    )


def _find_quant_file(quant_dir: Path, hemi: str, candidates: Sequence[str]) -> Path:
    for cand in candidates:
        p = quant_dir / f"{hemi}.{cand}.txt"
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"none of {[f'{hemi}.{c}.txt' for c in candidates]} found in {quant_dir}"
    )


def _load_hemisphere_props(quant_dir: Path, hemi: str) -> dict[str, np.ndarray]:
    """Load the 6 per-vertex measures for ``hemi`` from ``quant_dir``."""
    spec = {
        "H": ("mid.H",),
        "S": ("mid.S",),
        "Depth": ("gray.SD", "mid.SD"),
        "Thickness": ("thickness", "mid.CT"),
        "SCI": ("mid.SCI",),
        "LGI": ("gray.LGI", "mid.LGI"),
    }
    out: dict[str, np.ndarray] = {}
    for key, cands in spec.items():
        p = _find_quant_file(quant_dir, hemi, cands)
        out[key] = np.loadtxt(p, dtype=np.float64)
    return out


def remesh_stat_maps(
    quant_dir: str | Path,
    reg_dir: str | Path,
    out_dir: str | Path,
    *,
    uniform_sphere: str | Path,
    hemis: Sequence[str] = ("lh", "rh"),
    force: bool = False,
    legacy_names: bool = True,
) -> dict[str, dict[str, Path]]:
    """Resample ``H, S, SD, CT, LGI, SCI`` onto ``sphere_327680.vtk`` (spec §10 item 6, T40).

    Calls :func:`pybiscuit.sphere.remesh.resample` on ``<reg_dir>/<h>.sphere.reg.vtk`` and
    ``uniform_sphere`` (no ``--deg``).

    When ``legacy_names=True`` (default, matching ``biscuit.m:461-468``), writes into ``out_dir``:
    - ``<h>.mid.reg.{H,S,SCI}.txt``
    - ``<h>.gray.reg.{LGI,SD}.txt``
    - ``<h>.reg.thickness.txt``

    When ``legacy_names=False`` (matching ``surf_quant_stat_pipeline.m:266-301`` inside ``tmp4``),
    writes ``<h>.mid.reg.{H,S,SD,CT,LGI,SCI}.txt``.
    """
    quant_dir, reg_dir, out_dir = Path(quant_dir), Path(reg_dir), Path(out_dir)
    uniform_sphere = Path(uniform_sphere)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (measure_key, quant_candidates, legacy_stat_stem, scratch_stat_stem)
    map_specs = (
        ("H", ("mid.H",), "mid.reg.H", "mid.reg.H"),
        ("S", ("mid.S",), "mid.reg.S", "mid.reg.S"),
        ("SD", ("gray.SD", "mid.SD"), "gray.reg.SD", "mid.reg.SD"),
        ("CT", ("thickness", "mid.CT"), "reg.thickness", "mid.reg.CT"),
        ("LGI", ("gray.LGI", "mid.LGI"), "gray.reg.LGI", "mid.reg.LGI"),
        ("SCI", ("mid.SCI",), "mid.reg.SCI", "mid.reg.SCI"),
    )

    written: dict[str, dict[str, Path]] = {}
    for h in hemis:
        h_out: dict[str, Path] = {}
        written[h] = h_out
        todo_keys: list[tuple[str, Path]] = []
        todo_arrays: list[np.ndarray] = []
        for key, cands, leg_stem, scr_stem in map_specs:
            stem = leg_stem if legacy_names else scr_stem
            dst = out_dir / f"{h}.{stem}.txt"
            h_out[key] = dst
            if not force and dst.is_file():
                continue
            src = _find_quant_file(quant_dir, h, cands)
            arr = np.array(src.read_text().split(), dtype=np.float64).astype(np.float32)
            todo_keys.append((key, dst))
            todo_arrays.append(arr)
        if todo_arrays:
            sphere_reg = reg_dir / f"{h}.sphere.reg.vtk"
            resampled = resample(sphere_reg, uniform_sphere, todo_arrays)
            for (_, dst), vals in zip(todo_keys, resampled, strict=True):
                dst.write_bytes("".join(f"{x:f}\n" for x in vals.tolist()).encode())
    return written


def run_stats_pipeline(
    subject_or_out_dir: str | Path,
    *,
    quant_dir: str | Path | None = None,
    roi_dir: str | Path | None = None,
    curve_dir: str | Path | None = None,
    reg_dir: str | Path | None = None,
    stat_dir: str | Path | None = None,
    parc_root: str | Path,
    uniform_sphere: str | Path | None = None,
    protocols: Sequence[str] = ("braincolor", "dkt31"),
    remesh_maps: bool = True,
    force: bool = False,
) -> dict[str, StatsTable]:
    """Run Phase 4 (``surf_quant_stat_pipeline.m`` + ``biscuit.m:411-470``) across ``protocols``.

    Writes ``<stat_dir>/stats.<protocol>.csv`` for each protocol and (when ``remesh_maps=True``
    and ``uniform_sphere`` is given) the 12 registered-space property maps in ``<stat_dir>/``.
    """
    base = Path(subject_or_out_dir)
    q_dir = Path(quant_dir) if quant_dir is not None else base / "quant"
    r_dir = Path(roi_dir) if roi_dir is not None else base / "roi"
    c_dir = Path(curve_dir) if curve_dir is not None else base / "curve"
    g_dir = Path(reg_dir) if reg_dir is not None else base / "reg"
    s_dir = Path(stat_dir) if stat_dir is not None else base / "stat"
    parc_root = Path(parc_root)
    s_dir.mkdir(parents=True, exist_ok=True)

    props_lh = _load_hemisphere_props(q_dir, "lh")
    props_rh = _load_hemisphere_props(q_dir, "rh")
    curve_lh = read_bary(c_dir / "lh.gray.scurve.bary", unique=True)
    curve_rh = read_bary(c_dir / "rh.gray.scurve.bary", unique=True)

    tables: dict[str, StatsTable] = {}
    for proto in protocols:
        parc_lh = np.loadtxt(r_dir / f"lh.parc.{proto}.txt", dtype=np.int64)
        parc_rh = np.loadtxt(r_dir / f"rh.parc.{proto}.txt", dtype=np.int64)
        label_csv = parc_root / proto / "label.csv"
        tbl = compute_roi_and_curve_stats(
            props_lh,
            props_rh,
            parc_lh,
            parc_rh,
            curve_lh,
            curve_rh,
            label_csv,
        )
        write_stats_csv(s_dir / f"stats.{proto}.csv", tbl)
        tables[proto] = tbl

    if remesh_maps:
        if uniform_sphere is None:
            raise ValueError("uniform_sphere is required when remesh_maps=True")
        remesh_stat_maps(
            q_dir,
            g_dir,
            s_dir,
            uniform_sphere=uniform_sphere,
            force=force,
            legacy_names=True,
        )

    return tables
