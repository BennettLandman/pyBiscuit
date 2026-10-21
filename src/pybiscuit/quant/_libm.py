"""Platform-independent ``acosf``: the fdlibm single-precision algorithm of glibc's ``e_acosf.c``.

MeshLib weights face normals by ``acos`` of float32 dot products (``Mesh::updateNormal``); the C++
tools therefore see whatever ``acosf`` their libm returns.  Apple's and glibc's differ by 1 ulp on
~8 % of inputs, and the anisotropic LGI kernels flip kernel membership for ~3 % of vertices under
a 1-ulp change of the normals (see docs/reports/lgi-hull-2026-10-20.md).  Reproducing the glibc
algorithm (the reference image's libm; also what current Linux CI runners use) removes the
dependence on the host libm.  Original: Copyright (C) 1993 Sun Microsystems, Inc.; permission to
use, copy, modify and distribute is freely granted provided this notice is preserved.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

F = np.float32
_ONE = F(1.0)
_PI = F(3.1415925026e00)
_PIO2_HI = F(1.5707962513e00)
_PIO2_LO = F(7.5497894159e-08)
_PS0, _PS1, _PS2 = F(1.6666667163e-01), F(-3.2556581497e-01), F(2.0121252537e-01)
_PS3, _PS4, _PS5 = F(-4.0055535734e-02), F(7.9153501429e-04), F(3.4793309169e-05)
_QS1, _QS2, _QS3, _QS4 = (
    F(-2.4033949375e00),
    F(2.0209457874e00),
    F(-6.8828397989e-01),
    F(7.7038154006e-02),
)
_HALF = F(0.5)
_TWO = F(2.0)


@njit(cache=True, error_model="numpy")
def acosf(x):
    """float32 ``acos`` (bit-for-bit glibc <= 2.40 / fdlibm); ``x`` is a float32 scalar."""
    b = np.empty(1, dtype=np.float32)
    b[0] = x
    hx = b.view(np.int32)[0]
    ix = hx & 0x7FFFFFFF
    if ix == 0x3F800000:
        if hx > 0:
            return F(0.0)
        return _PI + _TWO * _PIO2_LO
    if ix > 0x3F800000:
        return F(np.nan)
    if ix < 0x3F000000:
        if ix <= 0x23000000:
            return _PIO2_HI + _PIO2_LO
        z = x * x
        p = z * (_PS0 + z * (_PS1 + z * (_PS2 + z * (_PS3 + z * (_PS4 + z * _PS5)))))
        q = _ONE + z * (_QS1 + z * (_QS2 + z * (_QS3 + z * _QS4)))
        r = p / q
        return _PIO2_HI - (x - (_PIO2_LO - x * r))
    if hx < 0:
        z = (_ONE + x) * _HALF
        p = z * (_PS0 + z * (_PS1 + z * (_PS2 + z * (_PS3 + z * (_PS4 + z * _PS5)))))
        q = _ONE + z * (_QS1 + z * (_QS2 + z * (_QS3 + z * _QS4)))
        s = F(math.sqrt(z))
        r = p / q
        w = r * s - _PIO2_LO
        return _PI - _TWO * (s + w)
    z = (_ONE - x) * _HALF
    s = F(math.sqrt(z))
    b[0] = s
    b.view(np.int32)[0] = b.view(np.int32)[0] & np.int32(-4096)  # 0xfffff000
    df = b[0]
    c = (z - df * df) / (s + df)
    p = z * (_PS0 + z * (_PS1 + z * (_PS2 + z * (_PS3 + z * (_PS4 + z * _PS5)))))
    q = _ONE + z * (_QS1 + z * (_QS2 + z * (_QS3 + z * _QS4)))
    r = p / q
    w = r * s + c
    return _TWO * (df + w)
