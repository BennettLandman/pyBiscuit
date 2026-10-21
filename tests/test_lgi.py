"""Gyrification port vs upstream oracle goldens (plan row B9.4).

Goldens: ``tests/data/golden/gyrification/<case>/`` from the non-FMA Gyrification oracle
(``scripts/oracles/make_golden_gyrification.py``); inputs are shared in ``.../inputs``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from _compare import assert_close_float32
from pybiscuit.io import read_vtk
from pybiscuit.quant.lgi import gyrification, read_curves, read_geodesic_map

G = Path(__file__).parent / "data" / "golden" / "gyrification"
INP = G / "inputs"
CASES = sorted(p.name for p in G.iterdir() if p.is_dir() and p.name != "inputs")


def _kwargs(argv):
    kw = {"sulcus": INP / "surface.scurve.bary", "gyrus": INP / "surface.gcurve.bary"}
    opts = {
        "-m": ("max_area", float), "-t": ("intv_area", float), "--speed": ("speed", float),
        "--refHullArea": ("ref_hull_area", float), "--refCortexArea": ("ref_cortex_area", float),
    }  # fmt: skip
    a = argv[1:]
    i = 0
    while i < len(a):
        o = a[i]
        if o in opts:
            kw[opts[o][0]] = opts[o][1](a[i + 1])
            i += 2
        elif o == "-s":
            kw["sulcus"] = INP / a[i + 1]
            i += 2
        elif o == "-g":
            kw["gyrus"] = INP / a[i + 1]
            i += 2
        elif o == "--inputMap":
            kw["input_map"] = G / "ratio1" / "gmap.txt"
            i += 2
        elif o == "-c":
            kw["corr"] = np.loadtxt(INP / a[i + 1], dtype=int)
            i += 2
        elif o in ("-i", "--outer", "-o", "--geodesicMap"):
            i += 2
        else:
            i += 1  # --kArea1/2
    return kw


@pytest.fixture(scope="module")
def meshes():
    return read_vtk(INP / "surface.vtk"), read_vtk(INP / "hull.vtk")


@pytest.mark.parametrize("case", CASES)
def test_gyrification_matches_oracle(case, meshes):
    cmd = json.loads((G / case / "command.json").read_text())
    res = gyrification(*meshes, **_kwargs(cmd["argv"]))
    labels = res.file_labels
    for name in cmd["outputs"]:
        gold = np.loadtxt(G / case / name)
        if name.startswith("lgi.lgi.map."):
            t = labels.index(int(name.split(".")[3]))
            assert_close_float32(res.lgi[:, t], gold, label=f"{case}:{name}")
        elif name.startswith("lgi.k1.map."):
            assert_close_float32(res.k1[:, labels.index(int(name.split(".")[3]))], gold, label=name)
        elif name.startswith("lgi.k2.map."):
            assert_close_float32(res.k2[:, labels.index(int(name.split(".")[3]))], gold, label=name)
        elif name == "gmap.txt":
            assert_close_float32(res.adj_dist, gold[:, 0], label="geodesic distance")
            assert np.array_equal(res.source, gold[:, 1].astype(int)), "nearest-source ids differ"
    assert sorted(labels) == sorted(
        int(n.split(".")[3]) for n in cmd["outputs"] if n.startswith("lgi.lgi.")
    )


def test_readers():
    plain = read_curves(INP / "surface.scurve")
    bary = read_curves(INP / "surface.scurve.bary")
    assert len(bary) == int((INP / "surface.scurve.bary").read_text().split()[0])
    assert {int(v) for c in plain for v in c} >= {int(v) for c in bary for v in c}


def test_input_map_roundtrip(meshes):
    d, s = read_geodesic_map(G / "ratio1" / "gmap.txt", meshes[0].n_vertices)
    assert d.shape == s.shape == (meshes[0].n_vertices,)


# ---------------------------------------------------------------- analytic diagnostics (no oracle)
def _curve_ids(n, k=24, seed=7):
    return np.sort(np.random.default_rng(seed).choice(n, size=k, replace=False))


def test_convex_sphere_lgi_is_one():
    from fixtures.lgi import fold_case

    surf, _ = fold_case(a=0.0, noise=0.0, hull_scale=1.0)
    res = gyrification(
        surf, surf, curve_ids=_curve_ids(surf.n_vertices), ref_hull_area=4 * np.pi * 50**2
    )
    assert np.allclose(res.lgi, 1.0, atol=1e-12)


def test_lgi_increases_with_fold_amplitude():
    from fixtures.lgi import fold_case

    med = []
    for a in (0.03, 0.08, 0.14, 0.2):
        surf, hull = fold_case(a=a)
        res = gyrification(surf, hull, curve_ids=_curve_ids(surf.n_vertices), ref_hull_area=20000.0)
        assert np.isfinite(res.lgi).all() and res.lgi.min() > 0
        med.append(float(np.median(res.lgi)))
    assert all(b > a for a, b in zip(med, med[1:], strict=False)), med


def test_heap_variants_agree_without_ties(meshes):
    kw = _kwargs(["x", "-m", "316", "--refHullArea", "20082.166"])
    a = gyrification(*meshes, **kw)
    b = gyrification(*meshes, heap="libstdcxx", **kw)
    assert_close_float32(b.lgi, a.lgi, label="libstdc++ vs libc++ heap order")


def test_open_mesh_rejected():
    from fixtures.meshes import cut_sphere

    m = cut_sphere(3)
    with pytest.raises(ValueError, match="closed"):
        gyrification(m, m, curve_ids=np.array([0, 5]))


def test_cli_matches_golden(tmp_path):
    from pybiscuit.cli import main

    out = tmp_path / "lgi"
    rc = main(
        ["lgi", "-i", str(INP / "surface.vtk"), "--outer", str(INP / "hull.vtk"),
         "-s", str(INP / "surface.scurve.bary"), "-g", str(INP / "surface.gcurve.bary"),
         "-o", str(out), "-m", "316", "--refHullArea", "20082.166", "--geodesicMap",
         str(tmp_path / "g.txt"), "--kArea1"]
    )  # fmt: skip
    assert rc == 0
    assert_close_float32(
        np.loadtxt(f"{out}.lgi.map.316.txt"), np.loadtxt(G / "ratio1" / "lgi.lgi.map.316.txt")
    )
    assert_close_float32(
        np.loadtxt(f"{out}.k1.map.316.txt"), np.loadtxt(G / "ratio1" / "lgi.k1.map.316.txt")
    )
    assert (tmp_path / "g.txt").read_text().splitlines()[0].count(" ") == 1


# ---------------------------------------------------------------- libm independence
def test_acosf_matches_fdlibm_table():
    """Bit-for-bit vs the C fdlibm/glibc algorithm (table made by compiling
    scripts/oracles/patches/acosf-fdlibm/acosf.c), on every platform."""
    from numba import njit

    from pybiscuit.quant._libm import acosf

    @njit
    def vec(x):
        out = np.empty_like(x)
        for i in range(x.size):
            out[i] = acosf(x[i])
        return out

    t = np.load(G.parent / "libm" / "acosf_fdlibm.npz")
    assert np.array_equal(vec(t["x"]).view(np.int32), t["y"].view(np.int32))


def test_libm_switch(meshes):
    """fdlibm is the default; ``host`` is available and agrees except at chaotic vertices."""
    kw = _kwargs(["x", "-m", "316", "--refHullArea", "20082.166"])
    a = gyrification(*meshes, **kw)
    b = gyrification(*meshes, libm="host", **kw)
    assert np.isfinite(b.lgi).all()
    assert np.median(np.abs(b.lgi - a.lgi) / a.lgi) < 1e-6  # almost all vertices are unaffected
    with pytest.raises(ValueError, match="libm"):
        gyrification(*meshes, libm="musl", **kw)


def test_no_out_of_bounds_access_without_jit():
    """Regression: the isotropic march read ``V1[cur]`` of a 1-row dummy array (access violation on
    Windows, silent elsewhere).  Numba does not bounds-check, so run the marching kernels as plain
    Python (indexing is checked) on a small mesh in a subprocess."""
    import os
    import subprocess
    import sys

    code = (
        "import numpy as np\n"
        "from fixtures.meshes import icosphere\n"
        "from pybiscuit.quant import _fastmarch as fm\n"
        "m = icosphere(2, 50.0)\n"
        "gw = fm.build_gw_mesh(m.vertices, m.faces)\n"
        "d, s, f = fm.propagate_iso(gw, [0, 7, 30])\n"
        "assert (s == 2).all()\n"
        "n = m.n_vertices\n"
        "u = np.tile([0.0, 0.0, 1.0], (n, 1))\n"
        "T, V1, l1, l2 = fm.set_tensor(gw, u, np.full(n, 0.5), np.full(n, 2.0))\n"
        "a = np.full(n, 3.0)\n"
        "k1 = np.zeros((n, 1)); k2 = np.zeros((n, 1))\n"
        "fm.kernel_range(0, gw.pos, gw.faces, gw.fnbr, gw.vf_ptr, gw.vf_idx, gw.ring_ptr,"
        " gw.ring_idx,"
        " gw.normal, T, V1, l1, l2, a, a, a, np.zeros(1, dtype=np.int64), False, 100.0,"
        " np.array([100.0]), 0, n, k1, k2)\n"
        "assert np.isfinite(k1).all() and (k1 > 0).all()\n"
    )
    env = dict(os.environ, NUMBA_DISABLE_JIT="1", PYTHONPATH=os.pathsep.join(sys.path))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-1500:]
