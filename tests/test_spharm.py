"""B6.2: MeshLib real spherical-harmonic basis (SphericalHarmonics::basis)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import scipy.special as sps

from _compare import assert_close_float32, assert_close_float64
from fixtures.sh_points import sh_fixture_points
from pybiscuit.sphere.spharm import (
    ShDeformation,
    basis,
    basis_size,
    legendre_schmidt,
    read_sh_coefficients,
    sh_index,
    synthesize,
    write_sh_coefficients,
)

ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
HARNESS = ORACLES / "meshlib-sh" / "harness_sh"
GOLDEN = Path(__file__).parent / "data" / "golden" / "meshlib"

# Metric: |a-b| <= ATOL + RTOL*|b| elementwise.  float32 outputs: measured max abs
# diff 0 (99.999% bitwise equal; rest 1 ulp = 6e-8 rel); float64: 1.3e-13 abs.
# F64 keeps the 1e-9 rel bound set before brief 03: it is looser than the 1e-10 policy because
# MeshLib's sqrt(1 - cos^2 t) amplifies a 1-ulp libm difference near the poles (see report b03).
F32 = dict(rel=1e-5, abs_=1e-7)
F64 = dict(rel=1e-9, abs_=1e-11)


def _sph(p):
    th = np.arctan2(np.hypot(p[:, 0], p[:, 1]), p[:, 2])  # well-conditioned near poles
    return th, np.arctan2(p[:, 1], p[:, 0])


def test_sizes_and_index():
    assert basis_size(20) == 441
    assert basis_size(9, 5) == 100 - 25
    assert sh_index(0, 0) == 0 and sh_index(1, -1) == 1 and sh_index(1, 1) == 3
    assert sh_index(5, -5, 5) == 0


def test_analytic_low_degree():
    # MeshLib convention: no Condon-Shortley phase, sin at -m, cos at +m.
    p = np.array([[0.3, -0.5, 0.8124038404635961]])
    p /= np.linalg.norm(p)
    x, y, z = p[0]
    Y = basis(p, 2)[0]
    c0 = 1 / np.sqrt(4 * np.pi)
    c1 = np.sqrt(3 / (4 * np.pi))
    c2 = np.sqrt(15 / (4 * np.pi))
    exp = [
        c0,
        c1 * y, c1 * z, c1 * x,  # m=-1,0,+1  (NO sign flip on x, unlike CS-phase tables)
        c2 * x * y,  # m=-2
        c2 * y * z,  # m=-1
        np.sqrt(5 / (16 * np.pi)) * (3 * z * z - 1),  # m=0
        c2 * x * z,  # m=+1
        c2 / 2 * (x * x - y * y),  # m=+2
    ]  # fmt: skip
    np.testing.assert_allclose(Y, exp, rtol=1e-12, atol=1e-14)


def test_matches_scipy_with_documented_convention():
    p = sh_fixture_points(300, 1).astype(np.float64)
    p /= np.linalg.norm(p, axis=1)[:, None]
    th, ph = _sph(p)
    Y = basis(p, 20)
    err = np.zeros(len(p))
    for ll in range(21):
        for m in range(ll + 1):
            c = sps.sph_harm_y(ll, m, th, ph)  # complex, WITH Condon-Shortley phase
            s = (-1.0) ** m
            if m == 0:
                e = abs(Y[:, sh_index(ll, 0)] - c.real)
            else:
                e = np.maximum(
                    abs(Y[:, sh_index(ll, m)] - np.sqrt(2) * s * c.real),
                    abs(Y[:, sh_index(ll, -m)] - np.sqrt(2) * s * c.imag),
                )
            err = np.maximum(err, e)
    # MeshLib derives sin(theta) as sqrt(1 - cos^2(theta)), losing precision within
    # ~1e-6 rad of the poles (faithfully ported; the oracle shows the same values).
    away = np.sin(th) > 1e-3
    assert err[away].max() < 1e-9
    assert err.max() < 1e-6


def test_orthonormal_quadrature():
    from pybiscuit.sphere.icosphere import icosphere

    # Gauss-Legendre in z x uniform phi: exact for degree <= 2*deg products.
    deg = 8
    xz, wz = np.polynomial.legendre.leggauss(2 * deg + 2)
    nphi = 4 * deg + 4
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    zz, pp = np.meshgrid(xz, phi, indexing="ij")
    r = np.sqrt(1 - zz**2)
    pts = np.stack([r * np.cos(pp), r * np.sin(pp), zz], -1).reshape(-1, 3)
    w = (wz[:, None] * np.ones(nphi) * 2 * np.pi / nphi).ravel()
    Y = basis(pts, deg)
    G = (Y * w[:, None]).T @ Y
    np.testing.assert_allclose(G, np.eye(Y.shape[1]), atol=1e-12)
    assert icosphere(1) is not None


def test_dtype_contract_and_from_degree():
    p = sh_fixture_points(50, 1)
    y32 = basis(p, 10)
    y64 = basis(p.astype(np.float64), 10)
    assert y32.dtype == np.float32 and y64.dtype == np.float64
    np.testing.assert_array_equal(y32, y64.astype(np.float32))  # float overload == rounded double
    part = basis(p, 10, from_degree=4)
    np.testing.assert_array_equal(part, y32[:, 16:])
    assert basis(p[0], 3).shape == (16,)


def test_scale_invariant_and_axis_conventions():
    p = sh_fixture_points(40, 0).astype(np.float64)
    np.testing.assert_allclose(basis(p * 100.0, 6), basis(p, 6), atol=1e-12)
    # z axis: only m=0 survive; z=0 -> elevation 0 (equator)
    Y = basis(np.array([[0, 0, 1.0]]), 4)[0]
    for ll in range(5):
        for m in range(-ll, ll + 1):
            want = np.sqrt((2 * ll + 1) / (4 * np.pi)) if m == 0 else 0.0
            assert abs(Y[sh_index(ll, m)] - want) < 1e-12


def test_legendre_schmidt_small():
    x = 0.3
    np.testing.assert_allclose(legendre_schmidt(0, x), [1])
    np.testing.assert_allclose(legendre_schmidt(1, x), [x, np.sqrt(1 - x * x)])
    got = legendre_schmidt(2, x)
    np.testing.assert_allclose(got[0], 0.5 * (3 * x * x - 1), rtol=1e-13)
    np.testing.assert_allclose(got[1], np.sqrt(3) * x * np.sqrt(1 - x * x), rtol=1e-13)
    np.testing.assert_allclose(got[2], np.sqrt(3) / 2 * (1 - x * x), rtol=1e-13)


@pytest.mark.parametrize("case", sorted(p.name for p in GOLDEN.glob("sh_*")))
def test_golden_vs_meshlib_oracle(case):
    d = GOLDEN / case
    cmd = json.loads((d / "command.json").read_text())
    z = np.load(d / "sh.npz")
    assert z["points"].dtype == np.float32
    if cmd["stored"] == "f32":
        got = basis(z["points"], cmd["degree"], cmd["from"])
        tol = F32
    else:
        got = basis(z["points"].astype(np.float64), cmd["degree"], cmd["from"])
        tol = F64
    assert got.dtype == z["y"].dtype
    (assert_close_float32 if cmd["stored"] == "f32" else assert_close_float64)(
        got, z["y"], label=case, **tol
    )


@pytest.mark.skipif(not HARNESS.exists(), reason="MeshLib SH oracle harness not built")
@pytest.mark.parametrize("frm", [0, 7])
def test_live_oracle_degree20(tmp_path, frm):
    pts = sh_fixture_points()  # 2000 random + specials + ico3
    pts.tofile(tmp_path / "p.bin")
    k = basis_size(20, frm)
    subprocess.run(
        [
            str(HARNESS),
            str(tmp_path / "p.bin"),
            str(len(pts)),
            "20",
            str(frm),
            str(tmp_path / "a"),
            str(tmp_path / "b"),
        ],  # fmt: skip
        check=True,
        capture_output=True,
    )
    o32 = np.fromfile(tmp_path / "a", np.float32).reshape(-1, k)
    o64 = np.fromfile(tmp_path / "b", np.float64).reshape(-1, k)
    assert_close_float32(basis(pts, 20, frm), o32, **F32)
    assert_close_float64(basis(pts.astype(np.float64), 20, frm), o64, **F64)


def test_coefficient_file_roundtrip_and_crop(tmp_path):
    rng = np.random.default_rng(1)
    n = 16
    coeff = rng.normal(size=(3, n)).astype(np.float32)
    dfm = ShDeformation(3, np.array([0.25, -1.5], np.float32), coeff)
    f = tmp_path / "c.txt"
    write_sh_coefficients(f, dfm)
    assert f.read_text().splitlines()[0] == "3"
    back = read_sh_coefficients(f)
    assert back.degree == 3 and back.coeff.dtype == np.float32
    np.testing.assert_allclose(back.coeff, dfm.coeff, atol=1e-6)
    crop = read_sh_coefficients(f, max_degree=1)
    assert crop.degree == 1 and crop.coeff.shape == (3, 4)
    np.testing.assert_array_equal(crop.coeff, back.coeff[:, :4])
    assert read_sh_coefficients(f, max_degree=9).degree == 3


def test_synthesize_matches_naive_and_truncation():
    p = sh_fixture_points(20, 1)
    deg = 5
    Y = basis(p, deg)
    rng = np.random.default_rng(2)
    c = rng.normal(size=(3, (deg + 1) ** 2)).astype(np.float32)
    full = synthesize(Y, c)
    np.testing.assert_allclose(full, Y @ c.T, rtol=2e-5, atol=2e-5)
    tr = synthesize(Y, c, degree=2)
    np.testing.assert_allclose(tr, Y[:, :9] @ c[:, :9].T, rtol=2e-5, atol=2e-5)
    assert synthesize(Y, c, degree=-0).shape == (len(p), 3)
    f64 = synthesize(Y.astype(np.float64), c.astype(np.float64))
    assert f64.dtype == np.float64
