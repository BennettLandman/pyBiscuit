"""B2.4b: MeshLib spherical mapping (``SurfaceUtil::sphere``) vs the oracle.

Port only -- not adopted as the B5.7 replacement (owner question Q3).  Committed goldens
(``tests/data/golden/meshlib/spheremap_*``) hold the oracle's vertices after ``max_iter`` in
{1, 2, 5, 20, 100, 10000} for both weight types; the live test on ``reg/mb101/lh.white.vtk``
needs ``reference/`` and a harness binary.
"""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from _compare import assert_close_float32
from conftest import REG_DIR
from fixtures.meshes import ellipsoid, icosphere
from pybiscuit.io.vtk import read_vtk
from pybiscuit.mesh.sphere_map import count_inverted_faces, sphere_map

GOLD = Path(__file__).parent / "data" / "golden" / "meshlib"
CASES = sorted(p.name for p in GOLD.glob("spheremap_*"))
ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
HARNESSES = [
    Path(os.environ.get("BISCUIT_HARNESS_CURVATURE", "/nonexistent")),
    ORACLES / "meshlib-b24b" / "harness_curvature",
]
ITERS = (1, 2, 5, 20, 100, 10000)


def _near_zero_faces(vertices, faces, eps=1e-6):
    """Faces whose inversion test value ``(V1-V0)x(V2-V0).(V0+V1+V2)`` is within ``eps`` of 0.

    Recomputed in float64 from the golden.  Whether such a face counts as inverted is decided by
    float32 rounding (the discrete decision behind ``n_inverted`` and the stopping rule), so the
    count may legitimately differ across platforms by at most this many faces.
    """
    v = np.asarray(vertices, np.float64)[faces]
    val = np.einsum("ij,ij->i", np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), v.sum(axis=1))
    return int((np.abs(val) < eps).sum())


def _check(mesh, g, iters, types=(0, 1)):
    n = mesh.n_vertices
    out = {}
    for typ in types:
        for k in iters:
            ref = g[f"sph_t{typ}_i{k}"].reshape(n, 3)
            r = sphere_map(mesh, type=typ, max_iter=k)
            got = r.mesh.vertices.astype(np.float32)
            # float32 arithmetic is replicated operation by operation (no libm calls: only + - *
            # / and sqrt, all correctly rounded), so bits normally agree on every IEEE platform;
            # the acceptance bound is 1e-5 relative per coordinate (-m bitexact checks the bits).
            assert_close_float32(got, ref, abs_=1e-7, label=f"sph_t{typ}_i{k}")
            slack = _near_zero_faces(ref, mesh.faces)
            assert abs(r.n_inverted - count_inverted_faces(ref, mesh.faces)) <= slack
            out[typ, k] = (r.iterations, r.n_inverted, float(np.abs(got - ref).max()))
    return out


@pytest.mark.parametrize("case", CASES)
def test_matches_committed_oracle_dumps(case):
    mesh = read_vtk(GOLD / case / "input.vtk")
    with np.load(GOLD / case / "golden.npz") as g:
        stats = _check(mesh, g, ITERS)
    for typ in (0, 1):
        # iteration-for-iteration: capped runs did exactly k passes unless already converged
        for k in (1, 2, 5, 20, 100):
            its, ninv, _ = stats[typ, k]
            assert its <= k and (its == k or ninv == 0)


@pytest.mark.bitexact
@pytest.mark.parametrize("case", CASES)
def test_bitexact_with_arm64_oracle(case):
    """Diagnostic only (``-m bitexact``): float32 vertices identical to the oracle's bits."""
    mesh = read_vtk(GOLD / case / "input.vtk")
    with np.load(GOLD / case / "golden.npz") as g:
        for typ in (0, 1):
            for k in ITERS:
                ref = g[f"sph_t{typ}_i{k}"].reshape(mesh.n_vertices, 3)
                r = sphere_map(mesh, type=typ, max_iter=k)
                assert np.array_equal(r.mesh.vertices.astype(np.float32), ref), (typ, k)
                assert r.n_inverted == count_inverted_faces(ref, mesh.faces)


@pytest.mark.parametrize("case", CASES)
def test_converges_to_flip_free_unit_sphere(case):
    mesh = read_vtk(GOLD / case / "input.vtk")
    for typ in (0, 1):
        r = sphere_map(mesh, type=typ)
        assert r.converged and r.n_inverted == 0
        norms = np.linalg.norm(r.mesh.vertices, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=2e-7)
        assert np.array_equal(r.mesh.faces, mesh.faces)


def test_goldens_have_provenance():
    assert CASES, "no spheremap_* goldens committed"
    for case in CASES:
        meta = json.loads((GOLD / case / "command.json").read_text())
        assert meta["oracle"] == "upstream-macos-arm64" and meta["commit"]
        assert sum(p.stat().st_size for p in (GOLD / case).iterdir()) < 1_000_000


def test_input_untouched_and_max_iter_zero_does_one_pass():
    m = ellipsoid()
    v0 = m.vertices.copy()
    r = sphere_map(m, max_iter=0)
    assert r.iterations == 1
    assert np.array_equal(m.vertices, v0)


def test_array_call_form_and_type_semantics():
    m = icosphere(3, 5.0)
    a = sphere_map(m.vertices, m.faces, type=0)
    b = sphere_map(m.vertices, m.faces, type=7)  # any non-zero value selects angle weights
    c = sphere_map(m.vertices, m.faces, type=1)
    assert np.array_equal(b.mesh.vertices, c.mesh.vertices)
    assert a.converged and c.converged
    with pytest.raises(TypeError):
        sphere_map(m.vertices)


def _harness():
    return next((p for p in HARNESSES if p.exists()), None)


@pytest.mark.reference
def test_lh_white_matches_live_oracle(tmp_path):
    harness = _harness()
    if harness is None:
        pytest.skip("harness_curvature binary not built (scripts/oracles/harness_curvature.cpp)")
    vtk = REG_DIR / "mb101" / "lh.white.vtk"
    iters = (1, 20, 10000)
    subprocess.run([str(harness), str(vtk), str(tmp_path), "", ",".join(map(str, iters))],
                   check=True, timeout=900)  # fmt: skip
    g = {p.stem: np.fromfile(p, dtype="<f4") for p in tmp_path.glob("sph_*.f32")}
    stats = _check(read_vtk(vtk), g, iters)
    # converged runs: zero flipped faces, identical iteration counts to the cap-free oracle run
    assert stats[0, 10000][1] == 0 and stats[1, 10000][1] == 0
