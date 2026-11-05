# ruff: noqa: E501
"""Tests for pybiscuit.mesh.interpolation (gw GW_TriangularInterpolation_*, row B2.7)."""

from __future__ import annotations

import ctypes
import math
import sys
import types

import numpy as np
import pytest

from conftest import REPO
from fixtures.interpolation_cases import CASES, build_mesh
from fixtures.meshes import icosphere
from pybiscuit.mesh import interpolation as interp
from pybiscuit.mesh.interpolation import (
    CubicInterpolation,
    InterpolationMesh,
    LinearInterpolation,
    QuadraticInterpolation,
    fit_2nd_order_polynomial_2d,
)

GOLD = REPO / "tests" / "data" / "golden" / "interpolation"
PERM = [(0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]
KIND = {"L": LinearInterpolation, "Q": QuadraticInterpolation, "C": CubicInterpolation}


def run_golden(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(port, oracle) rows ``value dx dy c0..c5`` for every query of a golden case."""
    m = build_mesh(CASES[name])
    dist = np.array([float(x) for x in (GOLD / name / "dist.txt").read_text().split()])
    im = InterpolationMesh(m.vertices, m.faces, dist)
    got = []
    for line in (GOLD / name / "queries.txt").read_text().splitlines():
        k, f, p, x, y = line.split()
        f, x, y = int(f), float(x), float(y)
        obj = KIND[k]()
        obj.set_up(im, f)
        v = [int(im.vert[f][i]) for i in PERM[int(p)]]
        dx, dy = obj.gradient(im, *v, x, y)
        coeffs = obj.coeffs if k != "L" else np.zeros(6)
        got.append([obj.value(im, *v, x, y), dx, dy, *coeffs])
    return np.array(got), np.loadtxt(GOLD / name / "out.txt")


def _apple_sincos():
    """``__sincos_stret`` from libSystem: what clang emits on macOS for a sin/cos pair."""

    class SC(ctypes.Structure):
        _fields_ = [("s", ctypes.c_double), ("c", ctypes.c_double)]

    fn = ctypes.CDLL("/usr/lib/libSystem.B.dylib").__sincos_stret
    fn.restype, fn.argtypes = SC, [ctypes.c_double]
    return fn


@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_vs_meshlib(name):
    """Value, gradient and fitted coefficients agree with the MeshLib harness to 1e-12 (relative)."""
    got, exp = run_golden(name)
    assert got.shape == exp.shape
    scale = np.maximum(np.abs(exp), 1.0)
    assert np.nanmax(np.abs(got - exp) / scale) <= 1e-12
    assert np.array_equal(np.isnan(got), np.isnan(exp))


@pytest.mark.skipif(sys.platform != "darwin", reason="needs Apple libm's __sincos_stret")
@pytest.mark.parametrize("name", sorted(CASES))
def test_golden_bit_identical_with_apple_sincos(name, monkeypatch):
    """With the oracle's fused sin/cos, every number is bit-identical (NaNs in the same places)."""
    fn = _apple_sincos()
    ns = types.SimpleNamespace(**{k: getattr(math, k) for k in dir(math) if not k.startswith("_")})
    ns.sin = lambda x: fn(x).s
    ns.cos = lambda x: fn(x).c
    monkeypatch.setattr(interp, "math", ns)
    got, exp = run_golden(name)
    same = (got == exp) | (np.isnan(got) & np.isnan(exp))
    assert same.all()


def test_open_boundary_fallbacks_give_nan_like_upstream():
    """A missing neighbour can push acos out of [-1, 1]; upstream then fits NaN coefficients."""
    got, exp = run_golden("cut_noisy_l3")
    assert np.isnan(exp).any()  # the golden really exercises it
    assert np.array_equal(np.isnan(got), np.isnan(exp))


def test_linear_reproduces_affine_field():
    """On a flat triangle an affine distance is reproduced exactly in value and gradient."""
    v = np.array([[2.0, 0, 0], [0, 3.0, 0], [0, 0, 0], [2.0, 3.0, 0]])
    f = np.array([[0, 1, 2], [1, 0, 3]])
    g = np.array([0.5, -1.0, 0.0])
    dist = v @ g + 7.0
    im = InterpolationMesh(v, f, dist)
    lin = LinearInterpolation()
    assert lin.value(im, 0, 1, 2, 0.25, 0.5) == pytest.approx(
        dist[0] * 0.25 + dist[1] * 0.5 + dist[2] * 0.25
    )
    dx, dy = lin.gradient(im, 0, 1, 2, 0.2, 0.2)
    e0 = (v[0] - v[2]) / np.linalg.norm(v[0] - v[2])
    e1 = (v[1] - v[2]) / np.linalg.norm(v[1] - v[2])
    assert dx * e0 + dy * e1 == pytest.approx(g)


def test_quadratic_reproduces_quadratic_field_on_flat_patch():
    """Six coplanar points: the fit is exact for a quadratic in the plane."""
    v = np.array(
        [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0], [1.0, 1.0, 0], [-1.0, 0.5, 0], [0.5, -1.0, 0]]
    )
    f = np.array([[0, 1, 2], [1, 0, 3], [1, 4, 2], [0, 2, 5]])

    def q(p):
        x, y = p[..., 0], p[..., 1]
        return 1 + 2 * x - y + 0.5 * x * y + 0.25 * x**2 - y**2

    im = InterpolationMesh(v, f, q(v))
    obj = im.set_up(0)
    assert isinstance(obj, QuadraticInterpolation)
    x, y = 0.3, 0.4
    p = x * v[0] + y * v[1] + (1 - x - y) * v[2]
    assert obj.value(im, 0, 1, 2, x, y) == pytest.approx(q(p), abs=1e-12)
    # the same point and value whichever vertex order the caller uses
    assert obj.value(im, 1, 2, 0, y, 1 - x - y) == pytest.approx(q(p), abs=1e-12)


def test_fit_matches_numpy_solve():
    rng = np.random.default_rng(3)
    pts, vals = rng.normal(size=(6, 2)), rng.normal(size=6)
    a = np.column_stack(
        [np.ones(6), pts[:, 0], pts[:, 1], pts[:, 0] * pts[:, 1], pts[:, 0] ** 2, pts[:, 1] ** 2]
    )
    assert np.allclose(
        fit_2nd_order_polynomial_2d(pts, vals), np.linalg.solve(a, vals), rtol=1e-10, atol=1e-12
    )


def test_cubic_is_the_quadratic_fit_and_reports_quadratic():
    m = icosphere(2, radius=10.0)
    im = InterpolationMesh(m, dist=np.linalg.norm(m.vertices - m.vertices[0], axis=1))
    q, c = QuadraticInterpolation(), CubicInterpolation()
    q.set_up(im, 5)
    c.set_up(im, 5)
    assert np.array_equal(q.coeffs, c.coeffs)
    assert c.get_type() == "quadratic"
    a, b, d = (int(i) for i in im.vert[5])
    assert q.gradient(im, a, b, d, 0.2, 0.3) == c.gradient(im, a, b, d, 0.2, 0.3)


def test_face_dispatch_follows_mesh_kind():
    m = icosphere(1)
    im = InterpolationMesh(m, dist=m.vertices[:, 2])
    assert isinstance(im.set_up(0), QuadraticInterpolation)  # GW default
    im.kind = "linear"
    lin = im.set_up(0)
    assert type(lin) is LinearInterpolation
    a, b, c = (int(i) for i in im.vert[0])
    assert im.gradient(0, a, b, c, 0.3, 0.3) == lin.gradient(im, a, b, c, 0.3, 0.3)
    im.kind = "cubic"
    with pytest.raises(ValueError, match="upstream assert"):
        im.set_up(0)
    with pytest.raises(ValueError):
        im.kind = "spline"
