#!/usr/bin/env python
"""Scale check of the Gyrification port against the oracle on a larger fold fixture (not in CI).

    .venv/bin/python scripts/validation/lgi_scale_check.py [level=6] [threads=4]

Writes the fold fixture (icosphere(level)), runs oracle CurveExtraction + Gyrification (nofma builds)
and the port on identical inputs; prints timings and relative differences of LGI / kernel areas.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from pybiscuit.io import read_vtk, write_vtk  # noqa: E402
from pybiscuit.mesh.geometry import vertex_areas  # noqa: E402
from pybiscuit.quant.lgi import gyrification  # noqa: E402
from tests.fixtures.lgi import fold_case  # noqa: E402

O = Path.home() / "dev" / "biscuit" / "oracles"  # noqa: E741
level = int(sys.argv[1]) if len(sys.argv) > 1 else 6
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 4
with tempfile.TemporaryDirectory() as td:
    w = Path(td)
    surf, hull = fold_case(level=level, l=14, m=6, noise=0.002)
    write_vtk(surf, w / "s.vtk", dtype="float")
    write_vtk(hull, w / "h.vtk", dtype="float")
    for fl in ("--sulcus", "--gyrus"):
        subprocess.run([str(O / "curve_extraction-nofma/bin/CurveExtraction"), "-i", "s.vtk", "-o", "s", fl,
                        "--noVtk", "--bary", "--sulcalSeed", "-0.01", "--gyralSeed", "0.01"], cwd=w, check=True,
                       capture_output=True)  # fmt: skip
    ha = float(vertex_areas(read_vtk(w / "h.vtk").vertices, read_vtk(w / "h.vtk").faces).sum())
    t = time.time()
    subprocess.run([str(O / "gyrification-nofma/bin/Gyrification"), "-i", "s.vtk", "--outer", "h.vtk", "-s",
                    "s.scurve.bary", "-g", "s.gcurve.bary", "-o", "o", "-m", "316", "--refHullArea", str(ha),
                    "--kArea1", "--kArea2"], cwd=w, check=True, capture_output=True)  # fmt: skip
    t_or = time.time() - t
    s2, h2 = read_vtk(w / "s.vtk"), read_vtk(w / "h.vtk")
    gyrification(s2, h2, sulcus=w / "s.scurve.bary", gyrus=w / "s.gcurve.bary", ref_hull_area=ha)  # JIT warm-up
    t = time.time()
    r = gyrification(s2, h2, sulcus=w / "s.scurve.bary", gyrus=w / "s.gcurve.bary", ref_hull_area=ha,
                     n_threads=threads)  # fmt: skip
    t_py = time.time() - t
    lab = int(r.deltas[0])
    for k, a in (("lgi", r.lgi), ("k1", r.k1), ("k2", r.k2)):
        g = np.loadtxt(w / f"o.{k}.map.{lab}.txt")
        rel = np.abs(a[:, 0] - g) / np.abs(g)
        print(f"{k}: max rel {rel.max():.2e}, n>1e-5: {(rel > 1e-5).sum()} of {g.size}")
    print(f"vertices {s2.n_vertices}; oracle {t_or:.1f} s (1 thread), port {t_py:.1f} s ({threads} threads)")
