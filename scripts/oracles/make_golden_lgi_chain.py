#!/usr/bin/env python
"""Golden generator for the TRACE -> LGI chain (plan row B20.1): the *oracle chain*.

    python scripts/oracles/make_golden_lgi_chain.py

Runs BISCUIT's calls (spec §8 and §9.3) with the no-FMA oracles on seeded fold fixtures: oracle
``CurveExtraction`` (sulcus on the pial stand-in, gyrus on the white stand-in, default seeds,
``--epRad 4 --nhDist 4 --lsTh 2.5 --bary --noVtk``), then oracle ``Gyrification`` with the ``lgi``
script's ``-m 316.000000 --speed 0.200000 --refHullArea 77100.000000``.  The hull is the fixture's
vertex-wise hull (the test correspondence approach of brief 07).  Output:
``tests/data/golden/lgi-chain/``.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]
from fixtures.lgi import SEED, fold_case  # noqa: E402
from pybiscuit.io import write_vtk  # noqa: E402
from pybiscuit.trace.pipeline import curve_argv  # noqa: E402

O = Path.home() / "dev" / "biscuit" / "oracles"  # noqa: E741
CE = O / "curve_extraction-nofma" / "bin" / "CurveExtraction"
GYR = O / "gyrification-nofma" / "bin" / "Gyrification"
OUT = REPO / "tests" / "data" / "golden" / "lgi-chain"
KW = dict(a=0.25, l=8, m=3, level=4, noise=0.004)  # noqa: C408  (default seeds find sulci and gyri)

with tempfile.TemporaryDirectory() as td:
    w = Path(td)
    gray, hull = fold_case(**KW)
    white, _ = fold_case(**KW, seed=SEED + 1)
    write_vtk(gray, w / "lh.gray.vtk", binary=False, dtype="float")
    write_vtk(white, w / "lh.white.vtk", binary=False, dtype="float")
    write_vtk(hull, w / "lh.gray.hull.corr.vtk", binary=False, dtype="float")
    cmds = []
    for kind, stem in (("sulcus", "gray"), ("gyrus", "white")):
        argv = curve_argv(kind, f"lh.{stem}.vtk", f"lh.{stem}")
        subprocess.run([str(CE), *argv], cwd=w, check=True, capture_output=True)
        cmds.append(["CurveExtraction", *argv])
    gargv = ["-i", "lh.gray.vtk", "-o", "lh.gray", "--outer", "lh.gray.hull.corr.vtk", "-s",
             "lh.gray.scurve.bary", "-g", "lh.white.gcurve.bary", "-m", "316.000000", "--speed",
             "0.200000", "--refHullArea", "77100.000000", "--nThreads", "1"]  # fmt: skip
    r = subprocess.run([str(GYR), *gargv], cwd=w, check=True, capture_output=True, text=True)
    cmds.append(["Gyrification", *gargv])
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    kept = ["lh.gray.vtk", "lh.white.vtk", "lh.gray.hull.corr.vtk", "lh.gray.scurve",
            "lh.gray.scurve.bary", "lh.white.gcurve", "lh.white.gcurve.bary"]  # fmt: skip
    kept += sorted(p.name for p in w.glob("lh.gray.lgi.map.*.txt"))
    for f in kept:
        shutil.copy(w / f, OUT / f)
    (OUT / "command.json").write_text(json.dumps({
        "tool": "CurveExtraction x2 + Gyrification (BISCUIT chain, spec 8 and 9.3)",
        "generator": "scripts/oracles/make_golden_lgi_chain.py", "fixture": KW,
        "commands": cmds, "oracle": "upstream-macos-arm64 (-ffp-contract=off), CE ded932d, Gyr b9fa760",
        "gyrification_stdout": r.stdout.strip().splitlines(), "files": kept}, indent=1))  # fmt: skip
    print(kept, r.stdout)
