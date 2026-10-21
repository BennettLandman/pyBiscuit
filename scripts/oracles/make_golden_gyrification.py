#!/usr/bin/env python
"""Golden generator for Gyrification (plan row B9.4).

    .venv/bin/python scripts/oracles/make_golden_gyrification.py [--list] [case ...|--all]

Inputs (shared by every case, committed once under ``tests/data/golden/gyrification/inputs/``): the
seeded fold fixture of ``tests/fixtures/lgi.py`` (surface + vertex-wise hull), and the curves written
by the *oracle* CurveExtraction (``curve_extraction-nofma``) on that surface.  Each case runs the
``-ffp-contract=off`` Gyrification oracle (``build_gyrification_nofma.sh``) and keeps its outputs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
GYR = ORACLES / "gyrification-nofma" / "bin" / "Gyrification"
CURVE = ORACLES / "curve_extraction-nofma" / "bin" / "CurveExtraction"
GOLDEN = REPO / "tests" / "data" / "golden" / "gyrification"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

HULL_AREA = None  # filled from the fixture (ratio 1 => kernel sizes are the nominal ones)
COMMON = [
    "-i",
    "surface.vtk",
    "--outer",
    "hull.vtk",
    "-s",
    "surface.scurve.bary",
    "-g",
    "surface.gcurve.bary",
]
OUTS = ["lgi.lgi.map.*.txt", "lgi.k1.map.*.txt", "lgi.k2.map.*.txt", "gmap.txt"]
CASES = {  # name -> extra args (HULL placeholder = fixture hull area)
    "defaults": ["--geodesicMap", "gmap.txt", "--kArea1", "--kArea2"],
    "ratio1": [
        "-m",
        "316",
        "--refHullArea",
        "HULL",
        "--geodesicMap",
        "gmap.txt",
        "--kArea1",
        "--kArea2",
    ],
    "intv": ["-m", "316", "-t", "100", "--refHullArea", "HULL", "--kArea1", "--kArea2"],
    "speed": ["-m", "200", "--speed", "0.5", "--refHullArea", "HULL", "--kArea1", "--kArea2"],
    "plain": ["-m", "316", "--refHullArea", "HULL", "PLAIN"],
    "inputmap": ["-m", "316", "--refHullArea", "HULL", "--inputMap", "ratio1_gmap.txt", "--kArea1"],
    "corr": ["-m", "316", "--refHullArea", "HULL", "-c", "corr.txt", "--kArea1", "--kArea2"],
}


def make_inputs(work: Path) -> float:
    from tests.fixtures.lgi import fold_case

    from pybiscuit.io import write_vtk
    from pybiscuit.mesh.geometry import vertex_areas

    surf, hull = fold_case()
    write_vtk(surf, work / "surface.vtk", binary=False, dtype="float")
    write_vtk(hull, work / "hull.vtk", binary=False, dtype="float")
    for flags in (["--sulcus"], ["--gyrus"]):
        subprocess.run(
            [str(CURVE), "-i", "surface.vtk", "-o", "surface", *flags, "--noVtk", "--bary",
             "--sulcalSeed", "-0.01", "--gyralSeed", "0.01"],
            cwd=work, check=True, capture_output=True,
        )  # fmt: skip
    # plain-format curves (second reader path): same run without --bary
    for flags in (["--sulcus"], ["--gyrus"]):
        subprocess.run(
            [str(CURVE), "-i", "surface.vtk", "-o", "surface", *flags, "--noVtk",
             "--sulcalSeed", "-0.01", "--gyralSeed", "0.01"],
            cwd=work, check=True, capture_output=True,
        )  # fmt: skip
    # corr: surface vertex -> outer vertex; every 7th vertex points at the previous one (many-to-one)
    n = surf.n_vertices
    corr = np.arange(n)
    corr[7::7] = corr[6:-1:7]
    np.savetxt(work / "corr.txt", corr, fmt="%d")
    hv = vertex_areas(hull.vertices.astype(np.float32), hull.faces)
    return float(hv.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        print("\n".join(CASES))
        return 0
    names = list(CASES) if a.all or not a.cases else a.cases
    GOLDEN.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        hull_area = make_inputs(work)
        inp = GOLDEN / "inputs"
        inp.mkdir(exist_ok=True)
        for f in ("surface.vtk", "hull.vtk", "corr.txt", "surface.scurve", "surface.gcurve",
                  "surface.scurve.bary", "surface.gcurve.bary"):  # fmt: skip
            shutil.copy(work / f, inp / f)
        (inp / "inputs.json").write_text(json.dumps(
            {"generator": "tests/fixtures/lgi.py:fold_case()", "hull_area": hull_area,
             "curve_oracle": "curve_extraction-nofma (ded932d)"}, indent=1))  # fmt: skip
        for name in names:
            extra = [
                str(round(hull_area, 3)) if x == "HULL" else x for x in CASES[name] if x != "PLAIN"
            ]
            if name == "inputmap":
                shutil.copy(GOLDEN / "ratio1" / "gmap.txt", work / "ratio1_gmap.txt")
            common = [x.removesuffix(".bary") for x in COMMON] if "PLAIN" in CASES[name] else COMMON
            argv = common + ["-o", "lgi"] + extra
            for f in [*work.glob("lgi.*"), work / "gmap.txt"]:
                f.unlink(missing_ok=True)
            subprocess.run([str(GYR), *argv], cwd=work, check=True, capture_output=True)
            out = GOLDEN / name
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir()
            kept = []
            for pat in OUTS:
                for f in sorted(work.glob(pat)):
                    shutil.copy(f, out / f.name)
                    kept.append(f.name)
            (out / "command.json").write_text(json.dumps({
                "tool": "gyrification", "case": name, "argv": ["$ORACLES/gyrification-nofma/bin/Gyrification", *argv],
                "commit": "b9fa760", "oracle": "upstream-macos-arm64 (-ffp-contract=off)",
                "inputs": "tests/data/golden/gyrification/inputs", "hull_area": hull_area, "outputs": kept}, indent=1))  # fmt: skip
            print(name, kept)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
