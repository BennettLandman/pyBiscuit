"""Real spherical-harmonic basis, port of MeshLib ``Util/SphericalHarmonics.cpp``.

Source: ilwoolyu/MeshLib @ dcef807, ``src/Util/SphericalHarmonics.cpp`` lines
16-139 (``SphericalHarmonics::basis``, float and double overloads),
``src/Util/Geom.cpp`` ``Coordinate::cart2sph`` (435-441) and
``Series::legendre2`` (1452-1534).  See ``docs/ports/spherical-harmonics.md``.

Convention (identical for MeshLib, HSD and SphericalRemesh)
-----------------------------------------------------------
* Row layout: degree-major.  For degree ``l`` and order ``m`` in ``[-l, l]`` the
  column is ``l*l + l + m - from_degree**2``.
* ``m == 0``:   ``sqrt((2l+1)/4pi) * P_l(cos t)``
* ``m  > 0``:   ``sqrt((2l+1)/4pi) * sqrt(2 (l-m)!/(l+m)!) * P_l^m(cos t) * cos(m phi)``
  at column ``center + m`` (``P_l^m`` WITHOUT the Condon-Shortley phase).
* ``m  < 0``:   same with ``sin(|m| phi)`` at column ``center - |m|``.
* ``phi = atan2(y, x)`` (0 if x=y=0), ``t = pi/2 - atan2(z, hypot(x, y))``
  (``Coordinate::cart2sph``; z==0 gives elevation 0).  The input is NOT
  normalised; only its direction matters.
* The float32 overload casts the input to double, computes everything in double
  and rounds each output to float32.  The float64 overload is the same code with
  no rounding.  Hence ``basis(p32).astype(f64)`` equals ``basis(p32.astype(f64))``
  rounded to float32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

__all__ = [
    "ShDeformation",
    "basis",
    "basis_size",
    "legendre_schmidt",
    "read_sh_coefficients",
    "sh_index",
    "synthesize",
    "write_sh_coefficients",
]

_PI = 3.141592653589793238462643
_TOL = 1.491668146240041e-154
_TSTART = 2.22044604925031308e-16


def basis_size(degree: int, from_degree: int = 0) -> int:
    """Number of basis functions for degrees ``from_degree..degree``."""
    return (degree + 1) ** 2 - from_degree**2


def sh_index(degree: int, order: int, from_degree: int = 0) -> int:
    """Column of Y_{degree,order} (order in [-degree, degree])."""
    return degree * degree + degree + order - from_degree**2


@njit(cache=True)
def _legendre2_schmidt(n, x, Y):  # pragma: no cover - exercised via tests
    """``Series::legendre2(n, x, Y, schmidt=true)``; fills Y[0..n] (double)."""
    if n < 0:
        return
    if n == 0:
        Y[0] = 1.0
        return
    factor = math.sqrt(1.0 - x * x)
    if n == 1:
        Y[0] = x
        Y[1] = factor
        return
    rootn = np.empty(2 * n + 1)
    for i in range(2 * n + 1):
        rootn[i] = math.sqrt(float(i))
    twocot = 0.0 if factor == 0 else -2.0 * x / factor
    sn = (-factor) ** n
    for i in range(n + 1):
        Y[i] = 0.0

    if factor > 0 and abs(sn) <= _TOL:
        v = 9.2 - math.log(_TOL) / (n * factor)
        w = 1.0 / math.log(v)
        m1 = 1.0 + n * factor * v * w * (1.0058 + w * (3.819 - w * 12.173))
        mm1 = min(n, int(math.floor(m1)))
        if x < 0:
            t = (n + 1) % 2
        else:
            t = mm1 % 2
        Y[mm1 - 1] = -_TSTART if t == 0 else _TSTART
        sumsq = _TOL
        for m in range(mm1 - 2, -1, -1):
            Y[m] = (
                (m + 1) * twocot * Y[m + 1] - rootn[n + m + 2] * rootn[n - m - 1] * Y[m + 2]
            ) / (rootn[n + m + 1] * rootn[n - m])
            sumsq += Y[m] * Y[m]
        scale = 1.0 / math.sqrt(2.0 * sumsq - Y[0] * Y[0])
        for i in range(mm1 + 1):
            Y[i] *= scale

    if x != 1 and abs(sn) >= _TOL:
        c = 1.0
        for i in range(1, n + 1):
            c *= 1.0 - 1.0 / (i * 2)
        Y[n] = math.sqrt(c) * sn
        Y[n - 1] = Y[n] * twocot * n / rootn[2 * n]
        for m in range(n - 2, -1, -1):
            Y[m] = (
                Y[m + 1] * twocot * (m + 1) - Y[m + 2] * rootn[n + m + 2] * rootn[n - m - 1]
            ) / (rootn[n + m + 1] * rootn[n - m])

    if factor == 0:
        Y[0] = x**n

    const1 = -1.0
    for j in range(1, n + 1):
        Y[j] *= rootn[2] * const1
        const1 *= -1.0


@njit(cache=True)
def _basis_point(degree, from_degree, px, py, pz, out):
    d = px * px + py * py
    phi = 0.0 if d == 0 else math.atan2(py, px)
    elev = 0.0 if pz == 0 else math.atan2(pz, math.sqrt(d))
    theta = _PI / 2 - elev
    ct = math.cos(theta)
    Pm = np.empty(degree + 1)
    for ll in range(from_degree, degree + 1):
        _legendre2_schmidt(ll, ct, Pm)
        lconst = math.sqrt((2 * ll + 1) / (4 * _PI))
        center = (ll + 1) * (ll + 1) - ll - 1 - from_degree * from_degree
        out[center] = lconst * Pm[0]
        for m in range(1, ll + 1):
            pre = lconst * Pm[m]
            out[center - m] = pre * math.sin(m * phi)
            out[center + m] = pre * math.cos(m * phi)


@njit(cache=True)
def _basis_all(points, degree, from_degree, out):
    for i in range(points.shape[0]):
        _basis_point(degree, from_degree, points[i, 0], points[i, 1], points[i, 2], out[i])


def legendre_schmidt(n: int, x: float) -> np.ndarray:
    """MeshLib ``Series::legendre2(n, x, Y, schmidt=true)``: ``Y[m]`` for m=0..n.

    Schmidt semi-normalised, no Condon-Shortley phase: for m>0
    ``Y[m] = sqrt(2 (n-m)!/(n+m)!) P_n^m(x)`` (CS phase removed), ``Y[0] = P_n(x)``.
    """
    out = np.zeros(max(n, 0) + 1)
    _legendre2_schmidt(int(n), float(x), out)
    return out


def basis(points, degree: int, from_degree: int = 0) -> np.ndarray:
    """Real SH basis ``Y`` at ``points`` (shape ``(N,3)`` or ``(3,)``).

    Returns ``(N, basis_size(degree, from_degree))`` (``(K,)`` for a single
    point).  float32 input -> float32 output (MeshLib ``float`` overload, the
    one SphericalRemesh calls); any other input -> float64 (the ``double``
    overload HSD calls).  Both compute in float64 internally.
    """
    p = np.asarray(points)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    if p.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if degree < 0 or from_degree < 0 or from_degree > degree + 1:
        raise ValueError("require 0 <= from_degree <= degree + 1")
    out_dtype = np.float32 if p.dtype == np.float32 else np.float64
    p64 = np.ascontiguousarray(p, dtype=np.float64)
    out = np.zeros((p64.shape[0], basis_size(degree, from_degree)), dtype=np.float64)
    _basis_all(p64, int(degree), int(from_degree), out)
    out = out.astype(out_dtype, copy=False)
    return out[0] if single else out


# ----------------------------------------------------------------------------
# Deformation coefficient files (HSD saveCoeff / SphericalRemesh -c)
# ----------------------------------------------------------------------------


@dataclass
class ShDeformation:
    """Contents of an HSD coefficient file.

    ``coeff`` has shape ``(3, (degree+1)**2)``: row 0 = rotation angle about the
    pole axis (``delta[0]``), rows 1/2 = tangent-plane displacement along
    ``tan1``/``tan2`` (``delta[1]``/``delta[2]``).  ``pole`` = (azimuth, elevation)
    in radians as written by HSD.
    """

    degree: int
    pole: np.ndarray
    coeff: np.ndarray


def read_sh_coefficients(path, max_degree: int = -1, dtype=np.float32) -> ShDeformation:
    """Parse an HSD coefficient file as SphericalRemeshing.cpp:100-147 does.

    Layout: line 1 ``degree``; line 2 ``pole0 pole1``; then
    ``(degree+1)**2`` lines ``c0 c1 c2``.  ``max_degree >= 0`` crops the degree
    (``--deg``: ``degree = min(degree, max_degree)``) and, like the C++ code, reads
    only the first ``(degree+1)**2`` rows.  ``dtype`` float32 mimics the
    ``%f`` reads of SphericalRemesh; float64 mimics HSD's ``%lf`` reads.
    """
    toks = Path(path).read_text().split()
    degree = int(toks[0])
    pole = np.array([float(toks[1]), float(toks[2])], dtype=dtype)
    if max_degree >= 0:
        degree = min(degree, max_degree)
    n = (degree + 1) ** 2
    vals = np.array(toks[3 : 3 + 3 * n], dtype=np.float64)
    if vals.size != 3 * n:
        raise ValueError("coefficient file truncated")
    coeff = np.ascontiguousarray(vals.reshape(n, 3).T, dtype=dtype)
    return ShDeformation(degree, pole, coeff)


def write_sh_coefficients(path, deformation: ShDeformation) -> None:
    """Write in HSD ``saveCoeff`` format (``%d``, ``%f %f``, ``%lf %lf %lf``)."""
    n = (deformation.degree + 1) ** 2
    lines = [
        f"{deformation.degree:d}",
        f"{float(deformation.pole[0]):f} {float(deformation.pole[1]):f}",
    ]
    c = deformation.coeff
    for i in range(n):
        lines.append(f"{float(c[0, i]):f} {float(c[1, i]):f} {float(c[2, i]):f}")
    Path(path).write_text("\n".join(lines) + "\n")


@njit(cache=True)
def _synth32(Y, coeff, n, stride, out):
    for p in range(Y.shape[0]):
        d0 = np.float32(0.0)
        d1 = np.float32(0.0)
        d2 = np.float32(0.0)
        for i in range(n):
            y = Y[p, i]
            d0 += y * coeff[0, i]
            d1 += y * coeff[1, i]
            d2 += y * coeff[2, i]
        out[p, 0] = d0
        out[p, 1] = d1
        out[p, 2] = d2


@njit(cache=True)
def _synth64(Y, coeff, n, stride, out):
    for p in range(Y.shape[0]):
        d0 = 0.0
        d1 = 0.0
        d2 = 0.0
        for i in range(n):
            y = Y[p, i]
            d0 += y * coeff[0, i]
            d1 += y * coeff[1, i]
            d2 += y * coeff[2, i]
        out[p, 0] = d0
        out[p, 1] = d1
        out[p, 2] = d2


def synthesize(Y, coeff, degree: int | None = None) -> np.ndarray:
    """Per-point ``delta[3]`` = sum_i Y[:, i] * coeff[k, i] for i < (degree+1)**2.

    Equals the first loop of ``SphericalRemeshing::reconsCoord`` (float32,
    sequential accumulation, Y stride = the file's degree) when ``Y`` and
    ``coeff`` are float32, and of ``HSD::updateCoordinate`` when float64.
    ``Y`` is ``(N, K)`` built with the *file* degree (K = (file_degree+1)**2);
    ``degree`` (default: file degree) truncates the sum, which is how
    ``--deg0..--deg`` produce progressively finer deformations.
    Returns ``(N, 3)`` of the same float dtype.
    """
    coeff = np.asarray(coeff)
    K = coeff.shape[1]
    n = K if degree is None else (min(degree, int(round(math.sqrt(K))) - 1) + 1) ** 2
    if Y.shape[1] != K:
        raise ValueError("Y must be built with the coefficient file's degree")
    f32 = Y.dtype == np.float32 and coeff.dtype == np.float32
    dt = np.float32 if f32 else np.float64
    Yc = np.ascontiguousarray(Y, dtype=dt)
    cc = np.ascontiguousarray(coeff, dtype=dt)
    out = np.zeros((Y.shape[0], 3), dtype=dt)
    (_synth32 if f32 else _synth64)(Yc, cc, n, K, out)
    return out
