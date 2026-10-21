"""Adapter for a PyMaCRUISE run directory (primary input route).

The run's ``run.json`` (schema_version 1) lists outputs as flat file records
(``path``, ``sha256``, ``bytes``, ``shape``, ``axis_codes``) without semantic keys,
so products are identified by stage and file name. Recorded paths are absolute at
build time; if the directory was moved, they are re-rooted at ``stages/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..model import InputError, InputFindings
from ._common import check_same_grid, make_product

__all__ = ["REQUIRED", "find_pymacruise", "is_pymacruise_run"]

# (stage, file name) -> (product, kind, frame)
_BY_STAGE_FILE: dict[tuple[str, str], tuple[str, str, str | None]] = {
    ("surface-thickness", "inner_surface.vtk"): ("inner_surface", "surface", "RAS"),
    ("surface-thickness", "central_surface.vtk"): ("central_surface", "surface", "RAS"),
    ("surface-thickness", "outer_surface.vtk"): ("outer_surface", "surface", "RAS"),
    ("surface-thickness", "inner_surface_thickness.vtk"): (
        "inner_surface_thickness", "surface", "RAS"),
    ("surface-thickness", "thickness_levelset.nii.gz"): ("thickness_levelset", "volume", "RAS"),
    ("surface-thickness", "thickness_summary.json"): ("thickness_summary", "table", None),
    ("cruise-core", "inner.nii.gz"): ("inner_levelset", "volume", "RAS"),
    ("cruise-core", "central.nii.gz"): ("central_levelset", "volume", "RAS"),
    ("cruise-core", "outer.nii.gz"): ("outer_levelset", "volume", "RAS"),
    ("maseg-refinement", "SEG_refine.nii.gz"): ("seg_refine", "volume", "RAS"),
}

REQUIRED = (
    "inner_surface", "central_surface", "outer_surface",
    "inner_levelset", "central_levelset", "outer_levelset",
    "seg_refine", "t1",
)  # fmt: skip

_REQUIRED_STAGES = ("surface-thickness", "cruise-core", "maseg-refinement")
_GRID_PRODUCTS = ["t1", "seg_refine", "inner_levelset", "central_levelset", "outer_levelset",
                  "thickness_levelset"]


def is_pymacruise_run(root: str | Path) -> bool:
    f = Path(root) / "run.json"
    if not f.is_file():
        return False
    try:
        d = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(d, dict) and "stages" in d and "final_outputs" in d and "run_signature" in d


def _rebase(recorded: str, root: Path) -> Path:
    """Resolve a recorded path; fall back to ``root/stages/...`` if the run was moved."""
    p = Path(recorded)
    if p.is_file():
        return p
    parts = p.parts
    for key in ("stages", "resources"):
        if key in parts:
            cand = root.joinpath(*parts[len(parts) - 1 - parts[::-1].index(key):])
            if cand.is_file():
                return cand
    return p


def _vtk_frame_ok(path: Path) -> str | None:
    """Return an error text unless the VTK header declares RAS mm (second line)."""
    with open(path, "rb") as fh:
        fh.readline()
        header = fh.readline().decode("ascii", "replace").strip()
    if "coordinate_frame=RAS" not in header or "units=mm" not in header:
        return f"{path.name}: header {header!r} does not declare coordinate_frame=RAS units=mm"
    return None


def find_pymacruise(
    root: str | Path, *, allow_incomplete: bool = False, verify_hashes: bool = True
) -> InputFindings:
    root = Path(root).resolve()
    run_json = root / "run.json"
    if not run_json.is_file():
        raise InputError(f"{root} is not a PyMaCRUISE run directory (no run.json)")
    try:
        run: dict[str, Any] = json.loads(run_json.read_text())
    except json.JSONDecodeError as exc:
        raise InputError(f"{run_json} is not valid JSON: {exc}") from exc

    f = InputFindings("pymacruise", str(root), "macruise", required=REQUIRED)
    stages = run.get("stages", {})
    f.source = {
        "schema_version": run.get("schema_version"),
        "status": run.get("status"),
        "converged": run.get("converged"),
        "run_signature": run.get("run_signature"),
        "implementation_signature": run.get("implementation_signature"),
        "pymacruise_version": run.get("pymacruise_version") or run.get("version"),
        "workflow_scope": run.get("workflow_scope"),
        "spatial_convention": run.get("spatial_convention"),
        "stages": {n: {"attempt": s.get("attempt"), "status": s.get("status")}
                   for n, s in stages.items()},
    }
    if f.source["pymacruise_version"] is None:
        f.warnings.append(
            "run.json records no PyMaCRUISE version; run_signature and implementation_signature "
            "are recorded instead"
        )

    problems: list[str] = []
    if run.get("schema_version") != 1:
        problems.append(f"unsupported run.json schema_version {run.get('schema_version')!r}")
    if run.get("spatial_convention") != "RAS":
        problems.append(f"spatial_convention is {run.get('spatial_convention')!r}, expected 'RAS'")
    if run.get("status") != "complete":
        problems.append(f"run status is {run.get('status')!r}, not 'complete'")
    if run.get("converged") is False:
        problems.append("run did not converge")
    for name in _REQUIRED_STAGES:
        st = stages.get(name)
        if st is None:
            problems.append(f"stage '{name}' is absent")
        elif st.get("status") != "complete":
            problems.append(f"stage '{name}' status is {st.get('status')!r}")
    if problems:
        if allow_incomplete:
            f.warnings += [f"(allowed) {p}" for p in problems]
        else:
            f.errors += [f"{p} (use --allow-incomplete to try anyway)" for p in problems]

    # products from each stage's recorded outputs, matched by stage + file name
    for (stage, fname), (name, kind, frame) in _BY_STAGE_FILE.items():
        rec = next(
            (o for o in stages.get(stage, {}).get("outputs", []) if Path(o["path"]).name == fname),
            None,
        )
        if rec is None:
            continue
        path = _rebase(rec["path"], root)
        if not path.is_file():
            f.errors.append(f"{name}: recorded output {rec['path']} is missing")
            continue
        extra = {"stage": stage, "attempt": stages[stage].get("attempt")}
        prod = make_product(name, path, kind, frame, findings=f,
                            recorded_sha256=rec.get("sha256") if verify_hashes else None, **extra)
        if prod is not None and kind == "surface":
            err = _vtk_frame_ok(path)
            if err:
                f.errors.append(err)

    # T1: the RAS-oriented copy PyMaCRUISE worked on (derived resource), else its input
    t1_rec = next((d for d in run.get("derived_resources", []) if d.get("role") == "t1"), None)
    src = "derived_resources[t1]"
    if t1_rec is None:
        t1_rec, src = run.get("inputs", {}).get("t1"), "inputs.t1"
    if t1_rec is None:
        f.errors.append("run.json records no T1")
    else:
        path = _rebase(t1_rec["path"], root)
        if not path.is_file():
            f.errors.append(f"t1: recorded file {t1_rec['path']} is missing")
        else:
            make_product("t1", path, "volume", "RAS", findings=f, source=src,
                         recorded_sha256=t1_rec.get("sha256") if verify_hashes else None)

    if not f.errors:
        check_same_grid(f, [n for n in _GRID_PRODUCTS if n in f.products])
    return f
