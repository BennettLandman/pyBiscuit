# ruff: noqa: E501
"""float32/float64 scalar helpers shared by the HSD kernels (MeshLib ``Geom.cpp`` semantics).

Everything mirrors the C++ operation order and types: MeshLib ``Vector`` is ``float[3]``,
``Coordinate::*`` float overloads, ``Statistics::*`` float overloads.  Numba is used without
``fastmath`` and without FMA contraction, so each expression below rounds exactly like the
x86-64 binary built from the same source.  Integer / Python-float literals are always wrapped
(``F(3)``) because numba would otherwise promote ``int + float32`` to float64.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from pybiscuit.sphere.aabb import _bary32, _cart2sph  # noqa: F401  (re-exported)
from pybiscuit.sphere.remesh import _cos, _cross3, _rot_point, _sin, _sph2cart, _unit3  # noqa: F401

F = np.float32
D = np.float64
PI = 3.141592653589793238462643

NJIT = {"cache": True, "error_model": "numpy"}


@njit(cache=True, inline="always")
def norm3(x, y, z):
    """``MathVector::norm`` (float32)."""
    return F(math.sqrt(D(x * x + y * y + z * z)))


@njit(cache=True, inline="always")
def dot3(ax, ay, az, bx, by, bz):
    """``MathVector::operator*`` (float32, left to right)."""
    return ax * bx + ay * by + az * bz


@njit(cache=True, inline="always")
def rotation_into(m, A, ax, ay, az, theta):
    """``Coordinate::rotation(axis, theta, mat)`` into preallocated float32 ``m`` (9), ``A`` (9)."""
    for i in range(9):
        m[i] = F(0)
    m[0] = F(1)
    m[4] = F(1)
    m[8] = F(1)
    n = norm3(ax, ay, az)
    if n == F(0):
        return
    ax, ay, az = _unit3(ax, ay, az)
    A[0] = F(0)
    A[1] = -az
    A[2] = ay
    A[3] = az
    A[4] = F(0)
    A[5] = -ax
    A[6] = -ay
    A[7] = ax
    A[8] = F(0)
    s = _sin(theta)
    for i in range(9):
        m[i] += A[i] * s
    omc = F(1) - _cos(theta)
    for i in range(9):
        for j in range(3):
            m[i] += A[j + (i // 3) * 3] * A[j * 3 + i % 3] * omc


@njit(cache=True, inline="always")
def bary32_snap(ax, ay, az, bx, by, bz, cx, cy, cz, px, py, pz, err):
    """``Coordinate::cart2bary(float*, ..., err)`` including the snapping branches (Geom.cpp 469-543)."""
    c0, c1, c2 = _bary32(ax, ay, az, bx, by, bz, cx, cy, cz, px, py, pz)
    if abs(c0) < err:
        c0 = F(0)
        c1 = F(1) - norm3(px - bx, py - by, pz - bz) / norm3(cx - bx, cy - by, cz - bz)
        c2 = F(1) - c1
        if abs(c1) < err:
            c1 = F(0)
            c2 = F(1)
        elif abs(c2) < err:
            c1 = F(1)
            c2 = F(0)
    elif abs(c1) < err:
        c1 = F(0)
        c2 = F(1) - norm3(px - cx, py - cy, pz - cz) / norm3(ax - cx, ay - cy, az - cz)
        c0 = F(1) - c2
        if abs(c2) < err:
            c2 = F(0)
            c0 = F(1)
        elif abs(c0) < err:
            c2 = F(1)
            c0 = F(0)
    elif abs(c2) < err:
        c2 = F(0)
        c0 = F(1) - norm3(px - ax, py - ay, pz - az) / norm3(bx - ax, by - ay, bz - az)
        c1 = F(1) - c0
        if abs(c0) < err:
            c0 = F(0)
            c1 = F(1)
        elif abs(c1) < err:
            c0 = F(1)
            c1 = F(0)
    return c0, c1, c2


@njit(cache=True, inline="always")
def face_ids(faces, i):
    return faces[i, 0], faces[i, 1], faces[i, 2]


@njit(cache=True)
def unit_rows_inplace(v):
    """Row-wise ``Vector::unit`` on float32 (N,3): ``v[i] /= norm`` when the norm is positive."""
    for i in range(v.shape[0]):
        x, y, z = _unit3(v[i, 0], v[i, 1], v[i, 2])
        v[i, 0] = x
        v[i, 1] = y
        v[i, 2] = z
