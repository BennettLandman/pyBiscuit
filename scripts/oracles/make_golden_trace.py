#!/usr/bin/env python
"""Write TRACE / geodesic / slicer goldens under tests/data/golden/{trace,geodesic,slicer}/.

    python scripts/oracles/make_golden_trace.py

Needs the no-FMA oracles from ``scripts/oracles/build_curve_extraction_nofma.sh``
(``oracles/curve_extraction-nofma/bin/{CurveExtraction,harness_geodesic,harness_slicer}``).
Inputs are regenerated at test time from ``tests/fixtures/trace_cases.py`` (not committed).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "src"))
from fixtures.meshes import folded_sphere  # noqa: E402
from fixtures.trace_cases import (  # noqa: E402
    RADIUS,
    TRACE_ARGS,
    TRACE_CASES,
    trace_mesh,
    write_ascii_vtk,
)  # fmt: skip

ORA = Path.home() / "dev" / "biscuit" / "oracles" / "curve_extraction-nofma" / "bin"
GOLD = REPO / "tests" / "data" / "golden"
META = {"oracle": "upstream-macos-arm64 (-ffp-contract=off)", "meshlib": "dcef807",
        "curve_extraction": "ded932d"}  # fmt: skip
GEO_QUERY_SEED = 20261020
GEO_FIXTURE = (0.15, 6, 3, 3)  # folded sphere level 3, radius 50
SLICE_FIXTURE = (0.15, 6, 3, 4)


def geo_queries(n: int) -> list[str]:
    rng = np.random.default_rng(GEO_QUERY_SEED)
    qs = []
    for _ in range(3):
        s, e = int(rng.integers(n)), int(rng.integers(n))
        qs += [f"D {s} 6.0", f"D {s} 1e9", f"E {s} {e}", f"P {s} {e}", f"P {e} {s}"]
    return qs


def slice_planes(v: np.ndarray, count: int = 60) -> np.ndarray:
    rng = np.random.default_rng(GEO_QUERY_SEED + 1)
    out = []
    for _ in range(count):
        nrm = rng.standard_normal(3)
        nrm /= np.linalg.norm(nrm)
        p = v[int(rng.integers(len(v)))] if rng.random() < 0.5 else (rng.random(3) - 0.5) * RADIUS
        out.append(np.float32([*nrm, -float(np.dot(p, nrm))]))
    return np.array(out, dtype=np.float32)


def main() -> None:
    for case in TRACE_CASES:
        d = GOLD / "trace" / case
        d.mkdir(parents=True, exist_ok=True)
        m = trace_mesh(case)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            write_ascii_vtk(t / "in.vtk", m.vertices, m.faces)
            argv = [str(ORA / "CurveExtraction"), "-i", "in.vtk", "-o", "out", *TRACE_ARGS]
            subprocess.run(argv, cwd=t, check=True, capture_output=True)
            for ext in ("scurve", "gcurve", "scurve.bary", "gcurve.bary", "spoint", "gpoint"):
                (d / f"out.{ext}").write_bytes((t / f"out.{ext}").read_bytes())
        a, l, mm, lev = TRACE_CASES[case]  # noqa: E741
        (d / "command.json").write_text(json.dumps(
            {"tool": "trace", "case": case, "argv": ["CurveExtraction", "-i", "in.vtk", "-o", "out",
             *TRACE_ARGS], "fixture": {"generator": "tests/fixtures/trace_cases.py", "a": a, "l": l,
             "m": mm, "level": lev, "radius": RADIUS}, **META}, indent=2))
    a, l, mm, lev = GEO_FIXTURE  # noqa: E741
    g = folded_sphere(a, l, mm, lev)
    v = (g.vertices * RADIUS).astype(np.float32)
    d = GOLD / "geodesic" / "fold_l3"
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        write_ascii_vtk(t / "in.vtk", v, g.faces)
        qs = geo_queries(len(v))
        (d / "queries.txt").write_text("\n".join(qs) + "\n")
        subprocess.run([str(ORA / "harness_geodesic"), "in.vtk", str(d / "queries.txt"),
                        str(d / "out.txt")], cwd=t, check=True)  # fmt: skip
    (d / "command.json").write_text(json.dumps(
        {"tool": "geodesic", "fixture": {"a": a, "l": l, "m": mm, "level": lev, "radius": RADIUS,
         "query_seed": GEO_QUERY_SEED}, **META}, indent=2))
    a, l, mm, lev = SLICE_FIXTURE  # noqa: E741
    g = folded_sphere(a, l, mm, lev)
    v = (g.vertices * RADIUS).astype(np.float32)
    d = GOLD / "slicer" / "fold_l4"
    d.mkdir(parents=True, exist_ok=True)
    planes = slice_planes(v)
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        write_ascii_vtk(t / "in.vtk", v, g.faces)
        (d / "planes.txt").write_text("".join(" ".join(f"{x:.9g}" for x in p) + "\n" for p in planes))
        subprocess.run([str(ORA / "harness_slicer"), "in.vtk", str(d / "planes.txt"),
                        str(d / "out.txt")], cwd=t, check=True)  # fmt: skip
    (d / "command.json").write_text(json.dumps(
        {"tool": "slicer", "fixture": {"a": a, "l": l, "m": mm, "level": lev, "radius": RADIUS,
         "plane_seed": GEO_QUERY_SEED + 1}, **META}, indent=2))


if __name__ == "__main__":
    main()
