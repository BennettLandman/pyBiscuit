# ruff: noqa: E501, E741
"""BOBYQA (Powell 2009), translated from the MIT-licensed C version in NLopt.

HSD's landmark pole search (``HSD::initTangentPlane``, HSD.cpp 831-864) calls
``min_bobyqa(2, delta, costFunc, xl, xu, 1e-1, 1e-6, m_maxIter)`` from ``src/bobyqa.h``.  That
header derives from Eric Thiebaut's C version, whose repository LICENSE says Powell's Fortran
BOBYQA was released under the LGPL, so it is not translated here (docs/clean-room.md,
docs/open-questions.md Q19).  This module is instead a line-by-line translation of
``src/algs/bobyqa/bobyqa.c`` from NLopt (stevengj/nlopt, master, fetched 2026-11-04), which is
the f2c translation of the same 2009 Powell Fortran.  Only the call interface of ``bobyqa.h``
was read (``npt = 2n+1``, ``iprint = 0``, ``maxfun = max_iter``).

Changes from NLopt's C, all to restore Powell's original stopping rules:

* NLopt's ``nlopt_stopping`` hooks become the plain ``maxfun`` limit (``nf >= maxfun``) at the
  same places; the forced-stop, time, ``minf_max`` and ``ftol`` tests (NLopt additions, inactive
  with NLopt's defaults) are dropped, as is SGJ's ``isinf(suma)`` singularity check.
* ``rhoend`` is passed in directly (NLopt derives it from ``xtol_rel``).
* No variable rescaling: NLopt rescales only when the initial steps differ between dimensions,
  and HSD's are equal.

The arithmetic keeps the f2c expression order on IEEE doubles (``np.float64``, no fused
multiply-add), so it reproduces a non-FMA build of Powell's algorithm.  Indices are the 1-based
f2c ones (``_P`` applies the pointer shifts of the C code).

----
Original license (NLopt ``src/algs/bobyqa/COPYRIGHT``):

Copyright (c) 2009 M. J. D. Powell (mjdp@cam.ac.uk)
Modifications Copyright (c) 2010 Massachusetts Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

D = np.float64

SUCCESS = 0
MAXFUN_REACHED = 1
ROUNDOFF_LIMITED = 2
RHOEND_REACHED = 3
INVALID_ARGS = -1


def _sqrt(x):
    return np.sqrt(D(x))


def _min2(a, b):
    return a if a <= b else b


def _max2(a, b):
    return a if a >= b else b


class _P:
    """C pointer after ``p -= shift``: ``p[i]`` is ``view[i - shift]`` (the view shares memory)."""

    __slots__ = ("a", "s")

    def __init__(self, a, s):
        self.a = a
        self.s = s

    def __getitem__(self, i):
        return self.a[i - self.s]

    def __setitem__(self, i, v):
        self.a[i - self.s] = v

    def ptr(self, i):
        """``&p[i]`` as a raw (unshifted) view."""
        return self.a[i - self.s :]


class _Stop:
    __slots__ = ("nf", "maxfun")

    def __init__(self, maxfun):
        self.nf = 0
        self.maxfun = int(maxfun)

    def evals(self) -> bool:
        return self.nf >= self.maxfun


def _update(n, npt, bmat_v, zmat_v, ndim, vlag_v, beta, denom, knew, w_v):
    zmat_dim1 = npt
    zmat = _P(zmat_v, 1 + zmat_dim1)
    bmat_dim1 = ndim
    bmat = _P(bmat_v, 1 + bmat_dim1)
    vlag = _P(vlag_v, 1)
    w = _P(w_v, 1)

    one = D(1.0)
    zero = D(0.0)
    nptm = npt - n - 1
    ztest = zero
    for k in range(1, npt + 1):
        for j in range(1, nptm + 1):
            ztest = _max2(ztest, abs(zmat[k + j * zmat_dim1]))
    ztest *= D(1e-20)

    # Apply the rotations that put zeros in the KNEW-th row of ZMAT.
    for j in range(2, nptm + 1):
        if abs(zmat[knew + j * zmat_dim1]) > ztest:
            d1 = zmat[knew + zmat_dim1]
            d2 = zmat[knew + j * zmat_dim1]
            temp = _sqrt(d1 * d1 + d2 * d2)
            tempa = zmat[knew + zmat_dim1] / temp
            tempb = zmat[knew + j * zmat_dim1] / temp
            for i in range(1, npt + 1):
                temp = tempa * zmat[i + zmat_dim1] + tempb * zmat[i + j * zmat_dim1]
                zmat[i + j * zmat_dim1] = (
                    tempa * zmat[i + j * zmat_dim1] - tempb * zmat[i + zmat_dim1]
                )
                zmat[i + zmat_dim1] = temp
        zmat[knew + j * zmat_dim1] = zero

    # Put the first NPT components of the KNEW-th column of HLAG into W.
    for i in range(1, npt + 1):
        w[i] = zmat[knew + zmat_dim1] * zmat[i + zmat_dim1]
    alpha = w[knew]
    tau = vlag[knew]
    vlag[knew] -= one

    # Complete the updating of ZMAT.
    temp = _sqrt(denom)
    tempb = zmat[knew + zmat_dim1] / temp
    tempa = tau / temp
    for i in range(1, npt + 1):
        zmat[i + zmat_dim1] = tempa * zmat[i + zmat_dim1] - tempb * vlag[i]

    # Finally, update the matrix BMAT.
    for j in range(1, n + 1):
        jp = npt + j
        w[jp] = bmat[knew + j * bmat_dim1]
        tempa = (alpha * vlag[jp] - tau * w[jp]) / denom
        tempb = (-beta * w[jp] - tau * vlag[jp]) / denom
        for i in range(1, jp + 1):
            bmat[i + j * bmat_dim1] = bmat[i + j * bmat_dim1] + tempa * vlag[i] + tempb * w[i]
            if i > npt:
                bmat[jp + (i - npt) * bmat_dim1] = bmat[i + j * bmat_dim1]


def _rescue(n, npt, xl_v, xu_v, stop, calfun, xbase_v, xpt_v, fval_v, xopt_v, gopt_v, hq_v,
            pq_v, bmat_v, zmat_v, ndim, sl_v, su_v, delta, kopt, vlag_v, ptsaux_v, ptsid_v, w_v):  # fmt: skip
    """Returns ``(rc, kopt)``."""
    zmat_dim1 = npt
    zmat = _P(zmat_v, 1 + zmat_dim1)
    xpt_dim1 = npt
    xpt = _P(xpt_v, 1 + xpt_dim1)
    xl = _P(xl_v, 1)
    xu = _P(xu_v, 1)
    xbase = _P(xbase_v, 1)
    fval = _P(fval_v, 1)
    xopt = _P(xopt_v, 1)
    gopt = _P(gopt_v, 1)
    hq = _P(hq_v, 1)
    pq = _P(pq_v, 1)
    bmat_dim1 = ndim
    bmat = _P(bmat_v, 1 + bmat_dim1)
    sl = _P(sl_v, 1)
    su = _P(su_v, 1)
    vlag = _P(vlag_v, 1)
    ptsaux = _P(ptsaux_v, 3)
    ptsid = _P(ptsid_v, 1)
    w = _P(w_v, 1)

    xp = D(0.0)
    xq = D(0.0)
    ihp = 0
    beta = D(0.0)
    denom = D(0.0)
    half = D(0.5)
    one = D(1.0)
    zero = D(0.0)
    np_ = n + 1
    sfrac = half / D(np_)
    nptm = npt - np_

    # Shift the interpolation points so that XOPT becomes the origin, and set ZMAT to zero.
    sumpq = zero
    winc = zero
    for k in range(1, npt + 1):
        distsq = zero
        for j in range(1, n + 1):
            xpt[k + j * xpt_dim1] -= xopt[j]
            d1 = xpt[k + j * xpt_dim1]
            distsq += d1 * d1
        sumpq += pq[k]
        w[ndim + k] = distsq
        winc = _max2(winc, distsq)
        for j in range(1, nptm + 1):
            zmat[k + j * zmat_dim1] = zero

    # Update HQ so that HQ and PQ define the second derivatives after the shift of XBASE.
    ih = 0
    for j in range(1, n + 1):
        w[j] = half * sumpq * xopt[j]
        for k in range(1, npt + 1):
            w[j] += pq[k] * xpt[k + j * xpt_dim1]
        for i in range(1, j + 1):
            ih += 1
            hq[ih] = hq[ih] + w[i] * xopt[j] + w[j] * xopt[i]

    # Shift XBASE, SL, SU and XOPT. Set BMAT to zero, and also set PTSAUX.
    for j in range(1, n + 1):
        xbase[j] += xopt[j]
        sl[j] -= xopt[j]
        su[j] -= xopt[j]
        xopt[j] = zero
        ptsaux[(j << 1) + 1] = _min2(delta, su[j])
        ptsaux[(j << 1) + 2] = _max2(-delta, sl[j])
        if ptsaux[(j << 1) + 1] + ptsaux[(j << 1) + 2] < zero:
            temp = ptsaux[(j << 1) + 1]
            ptsaux[(j << 1) + 1] = ptsaux[(j << 1) + 2]
            ptsaux[(j << 1) + 2] = temp
        if abs(ptsaux[(j << 1) + 2]) < half * abs(ptsaux[(j << 1) + 1]):
            ptsaux[(j << 1) + 2] = half * ptsaux[(j << 1) + 1]
        for i in range(1, ndim + 1):
            bmat[i + j * bmat_dim1] = zero
    fbase = fval[kopt]

    # Identifiers of the artificial interpolation points along the coordinate directions.
    ptsid[1] = sfrac
    for j in range(1, n + 1):
        jp = j + 1
        jpn = jp + n
        ptsid[jp] = D(j) + sfrac
        if jpn <= npt:
            ptsid[jpn] = D(j) / D(np_) + sfrac
            temp = one / (ptsaux[(j << 1) + 1] - ptsaux[(j << 1) + 2])
            bmat[jp + j * bmat_dim1] = -temp + one / ptsaux[(j << 1) + 1]
            bmat[jpn + j * bmat_dim1] = temp + one / ptsaux[(j << 1) + 2]
            bmat[j * bmat_dim1 + 1] = -bmat[jp + j * bmat_dim1] - bmat[jpn + j * bmat_dim1]
            zmat[j * zmat_dim1 + 1] = _sqrt(2.0) / abs(ptsaux[(j << 1) + 1] * ptsaux[(j << 1) + 2])
            zmat[jp + j * zmat_dim1] = zmat[j * zmat_dim1 + 1] * ptsaux[(j << 1) + 2] * temp
            zmat[jpn + j * zmat_dim1] = -zmat[j * zmat_dim1 + 1] * ptsaux[(j << 1) + 1] * temp
        else:
            bmat[j * bmat_dim1 + 1] = -one / ptsaux[(j << 1) + 1]
            bmat[jp + j * bmat_dim1] = one / ptsaux[(j << 1) + 1]
            d1 = ptsaux[(j << 1) + 1]
            bmat[j + npt + j * bmat_dim1] = -half * (d1 * d1)

    # Any remaining identifiers with their nonzero elements of ZMAT.
    if npt >= n + np_:
        for k in range(np_ << 1, npt + 1):
            iw = int((D(k - np_) - half) / D(n))
            ip = k - np_ - iw * n
            iq = ip + iw
            if iq > n:
                iq -= n
            ptsid[k] = D(ip) + D(iq) / D(np_) + sfrac
            temp = one / (ptsaux[(ip << 1) + 1] * ptsaux[(iq << 1) + 1])
            zmat[(k - np_) * zmat_dim1 + 1] = temp
            zmat[ip + 1 + (k - np_) * zmat_dim1] = -temp
            zmat[iq + 1 + (k - np_) * zmat_dim1] = -temp
            zmat[k + (k - np_) * zmat_dim1] = temp
    nrem = npt
    kold = 1
    knew = kopt

    label = 80
    while True:
        if label == 80:
            # Reorder the provisional points: exchange PTSID(KOLD) with PTSID(KNEW).
            for j in range(1, n + 1):
                temp = bmat[kold + j * bmat_dim1]
                bmat[kold + j * bmat_dim1] = bmat[knew + j * bmat_dim1]
                bmat[knew + j * bmat_dim1] = temp
            for j in range(1, nptm + 1):
                temp = zmat[kold + j * zmat_dim1]
                zmat[kold + j * zmat_dim1] = zmat[knew + j * zmat_dim1]
                zmat[knew + j * zmat_dim1] = temp
            ptsid[kold] = ptsid[knew]
            ptsid[knew] = zero
            w[ndim + knew] = zero
            nrem -= 1
            if knew != kopt:
                temp = vlag[kold]
                vlag[kold] = vlag[knew]
                vlag[knew] = temp
                _update(n, npt, bmat.ptr(1 + bmat_dim1), zmat.ptr(1 + zmat_dim1), ndim,
                        vlag.ptr(1), beta, denom, knew, w.ptr(1))  # fmt: skip
                if nrem == 0:
                    return SUCCESS, kopt
                for k in range(1, npt + 1):
                    w[ndim + k] = abs(w[ndim + k])
            label = 120
        elif label == 120:
            # Pick KNEW, an original point that has not yet replaced a provisional one.
            dsqmin = zero
            for k in range(1, npt + 1):
                if w[ndim + k] > zero:
                    if dsqmin == zero or w[ndim + k] < dsqmin:
                        knew = k
                        dsqmin = w[ndim + k]
            if dsqmin == zero:
                label = 260
                continue
            # Form the W-vector of the chosen original interpolation point.
            for j in range(1, n + 1):
                w[npt + j] = xpt[knew + j * xpt_dim1]
            for k in range(1, npt + 1):
                sum_ = zero
                if k == kopt:
                    pass
                elif ptsid[k] == zero:
                    for j in range(1, n + 1):
                        sum_ += w[npt + j] * xpt[k + j * xpt_dim1]
                else:
                    ip = int(ptsid[k])
                    if ip > 0:
                        sum_ = w[npt + ip] * ptsaux[(ip << 1) + 1]
                    iq = int(D(np_) * ptsid[k] - D(ip * np_))
                    if iq > 0:
                        iw = 1
                        if ip == 0:
                            iw = 2
                        sum_ += w[npt + iq] * ptsaux[iw + (iq << 1)]
                w[k] = half * sum_ * sum_
            # VLAG and BETA for reinstating XPT(KNEW,.).
            for k in range(1, npt + 1):
                sum_ = zero
                for j in range(1, n + 1):
                    sum_ += bmat[k + j * bmat_dim1] * w[npt + j]
                vlag[k] = sum_
            beta = zero
            for j in range(1, nptm + 1):
                sum_ = zero
                for k in range(1, npt + 1):
                    sum_ += zmat[k + j * zmat_dim1] * w[k]
                beta -= sum_ * sum_
                for k in range(1, npt + 1):
                    vlag[k] += sum_ * zmat[k + j * zmat_dim1]
            bsum = zero
            distsq = zero
            for j in range(1, n + 1):
                sum_ = zero
                for k in range(1, npt + 1):
                    sum_ += bmat[k + j * bmat_dim1] * w[k]
                jp = j + npt
                bsum += sum_ * w[jp]
                for ip in range(npt + 1, ndim + 1):
                    sum_ += bmat[ip + j * bmat_dim1] * w[ip]
                bsum += sum_ * w[jp]
                vlag[jp] = sum_
                d1 = xpt[knew + j * xpt_dim1]
                distsq += d1 * d1
            beta = half * distsq * distsq + beta - bsum
            vlag[kopt] += one
            # KOLD: the provisional point to delete (avoid a small UPDATE denominator).
            denom = zero
            vlmxsq = zero
            for k in range(1, npt + 1):
                if ptsid[k] != zero:
                    hdiag = zero
                    for j in range(1, nptm + 1):
                        d1 = zmat[k + j * zmat_dim1]
                        hdiag += d1 * d1
                    d1 = vlag[k]
                    den = beta * hdiag + d1 * d1
                    if den > denom:
                        kold = k
                        denom = den
                d3 = vlag[k]
                vlmxsq = _max2(vlmxsq, d3 * d3)
            if denom <= vlmxsq * D(0.01):
                w[ndim + knew] = -w[ndim + knew] - winc
                label = 120
                continue
            label = 80
        else:  # label 260
            for kpt in range(1, npt + 1):
                if ptsid[kpt] == zero:
                    continue
                if stop.evals():
                    return MAXFUN_REACHED, kopt
                ih = 0
                for j in range(1, n + 1):
                    w[j] = xpt[kpt + j * xpt_dim1]
                    xpt[kpt + j * xpt_dim1] = zero
                    temp = pq[kpt] * w[j]
                    for i in range(1, j + 1):
                        ih += 1
                        hq[ih] += temp * w[i]
                pq[kpt] = zero
                ip = int(ptsid[kpt])
                iq = int(D(np_) * ptsid[kpt] - D(ip * np_))
                if ip > 0:
                    xp = ptsaux[(ip << 1) + 1]
                    xpt[kpt + ip * xpt_dim1] = xp
                if iq > 0:
                    xq = ptsaux[(iq << 1) + 1]
                    if ip == 0:
                        xq = ptsaux[(iq << 1) + 2]
                    xpt[kpt + iq * xpt_dim1] = xq
                # VQUAD: the current model at the new point.
                vquad = fbase
                if ip > 0:
                    ihp = (ip + ip * ip) // 2
                    vquad += xp * (gopt[ip] + half * xp * hq[ihp])
                if iq > 0:
                    ihq = (iq + iq * iq) // 2
                    vquad += xq * (gopt[iq] + half * xq * hq[ihq])
                    if ip > 0:
                        iw = _max2(ihp, ihq) - abs(ip - iq)
                        vquad += xp * xq * hq[iw]
                for k in range(1, npt + 1):
                    temp = zero
                    if ip > 0:
                        temp += xp * xpt[k + ip * xpt_dim1]
                    if iq > 0:
                        temp += xq * xpt[k + iq * xpt_dim1]
                    vquad += half * pq[k] * temp * temp
                # F at the new interpolation point.
                for i in range(1, n + 1):
                    w[i] = _min2(_max2(xl[i], xbase[i] + xpt[kpt + i * xpt_dim1]), xu[i])
                    if xpt[kpt + i * xpt_dim1] == sl[i]:
                        w[i] = xl[i]
                    if xpt[kpt + i * xpt_dim1] == su[i]:
                        w[i] = xu[i]
                stop.nf += 1
                f = D(calfun(w.ptr(1)[:n].copy()))
                fval[kpt] = f
                if f < fval[kopt]:
                    kopt = kpt
                if stop.evals():
                    return MAXFUN_REACHED, kopt
                diff = f - vquad
                # Update the quadratic model.
                for i in range(1, n + 1):
                    gopt[i] += diff * bmat[kpt + i * bmat_dim1]
                for k in range(1, npt + 1):
                    sum_ = zero
                    for j in range(1, nptm + 1):
                        sum_ += zmat[k + j * zmat_dim1] * zmat[kpt + j * zmat_dim1]
                    temp = diff * sum_
                    if ptsid[k] == zero:
                        pq[k] += temp
                    else:
                        ip = int(ptsid[k])
                        iq = int(D(np_) * ptsid[k] - D(ip * np_))
                        ihq = (iq * iq + iq) // 2
                        if ip == 0:
                            d1 = ptsaux[(iq << 1) + 2]
                            hq[ihq] += temp * (d1 * d1)
                        else:
                            ihp = (ip * ip + ip) // 2
                            d1 = ptsaux[(ip << 1) + 1]
                            hq[ihp] += temp * (d1 * d1)
                            if iq > 0:
                                d1 = ptsaux[(iq << 1) + 1]
                                hq[ihq] += temp * (d1 * d1)
                                iw = _max2(ihp, ihq) - abs(iq - ip)
                                hq[iw] += temp * ptsaux[(ip << 1) + 1] * ptsaux[(iq << 1) + 1]
                ptsid[kpt] = zero
            return SUCCESS, kopt


def _altmov(n, npt, xpt_v, xopt_v, bmat_v, zmat_v, ndim, sl_v, su_v, kopt, knew, adelt,
            xnew_v, xalt_v, glag_v, hcol_v, w_v):  # fmt: skip
    """Returns ``(alpha, cauchy)``."""
    zmat_dim1 = npt
    zmat = _P(zmat_v, 1 + zmat_dim1)
    xpt_dim1 = npt
    xpt = _P(xpt_v, 1 + xpt_dim1)
    xopt = _P(xopt_v, 1)
    bmat_dim1 = ndim
    bmat = _P(bmat_v, 1 + bmat_dim1)
    sl = _P(sl_v, 1)
    su = _P(su_v, 1)
    xnew = _P(xnew_v, 1)
    xalt = _P(xalt_v, 1)
    glag = _P(glag_v, 1)
    hcol = _P(hcol_v, 1)
    w = _P(w_v, 1)

    ksav = 0
    step = D(0.0)
    csave = D(0.0)
    ibdsav = 0
    stpsav = D(0.0)
    half = D(0.5)
    one = D(1.0)
    zero = D(0.0)
    const = one + _sqrt(2.0)
    for k in range(1, npt + 1):
        hcol[k] = zero
    for j in range(1, npt - n - 1 + 1):
        temp = zmat[knew + j * zmat_dim1]
        for k in range(1, npt + 1):
            hcol[k] += temp * zmat[k + j * zmat_dim1]
    alpha = hcol[knew]
    ha = half * alpha

    # Gradient of the KNEW-th Lagrange function at XOPT.
    for i in range(1, n + 1):
        glag[i] = bmat[knew + i * bmat_dim1]
    for k in range(1, npt + 1):
        temp = zero
        for j in range(1, n + 1):
            temp += xpt[k + j * xpt_dim1] * xopt[j]
        temp = hcol[k] * temp
        for i in range(1, n + 1):
            glag[i] += temp * xpt[k + i * xpt_dim1]

    # Search for a large denominator along the lines through XOPT and another point.
    presav = zero
    for k in range(1, npt + 1):
        if k == kopt:
            continue
        dderiv = zero
        distsq = zero
        for i in range(1, n + 1):
            temp = xpt[k + i * xpt_dim1] - xopt[i]
            dderiv += glag[i] * temp
            distsq += temp * temp
        subd = adelt / _sqrt(distsq)
        slbd = -subd
        ilbd = 0
        iubd = 0
        sumin = _min2(one, subd)
        # Revise SLBD and SUBD because of the bounds in SL and SU.
        for i in range(1, n + 1):
            temp = xpt[k + i * xpt_dim1] - xopt[i]
            if temp > zero:
                if slbd * temp < sl[i] - xopt[i]:
                    slbd = (sl[i] - xopt[i]) / temp
                    ilbd = -i
                if subd * temp > su[i] - xopt[i]:
                    subd = _max2(sumin, (su[i] - xopt[i]) / temp)
                    iubd = i
            elif temp < zero:
                if slbd * temp > su[i] - xopt[i]:
                    slbd = (su[i] - xopt[i]) / temp
                    ilbd = i
                if subd * temp < sl[i] - xopt[i]:
                    subd = _max2(sumin, (sl[i] - xopt[i]) / temp)
                    iubd = -i
        if k == knew:
            diff = dderiv - one
            step = slbd
            vlag = slbd * (dderiv - slbd * diff)
            isbd = ilbd
            temp = subd * (dderiv - subd * diff)
            if abs(temp) > abs(vlag):
                step = subd
                vlag = temp
                isbd = iubd
            tempd = half * dderiv
            tempa = tempd - diff * slbd
            tempb = tempd - diff * subd
            if tempa * tempb < zero:
                temp = tempd * tempd / diff
                if abs(temp) > abs(vlag):
                    step = tempd / diff
                    vlag = temp
                    isbd = 0
        else:
            step = slbd
            vlag = slbd * (one - slbd)
            isbd = ilbd
            temp = subd * (one - subd)
            if abs(temp) > abs(vlag):
                step = subd
                vlag = temp
                isbd = iubd
            if subd > half:
                if abs(vlag) < D(0.25):
                    step = half
                    vlag = D(0.25)
                    isbd = 0
            vlag *= dderiv
        # PREDSQ for the current line search, and PRESAV.
        temp = step * (one - step) * distsq
        predsq = vlag * vlag * (vlag * vlag + ha * temp * temp)
        if predsq > presav:
            presav = predsq
            ksav = k
            stpsav = step
            ibdsav = isbd

    # Construct XNEW in a way that satisfies the bound constraints exactly.
    for i in range(1, n + 1):
        temp = xopt[i] + stpsav * (xpt[ksav + i * xpt_dim1] - xopt[i])
        xnew[i] = _max2(sl[i], _min2(su[i], temp))
    if ibdsav < 0:
        xnew[-ibdsav] = sl[-ibdsav]
    if ibdsav > 0:
        xnew[ibdsav] = su[ibdsav]

    # The constrained Cauchy step in W.
    bigstp = adelt + adelt
    iflag = 0
    cauchy = zero
    while True:  # L100
        wfixsq = zero
        ggfree = zero
        for i in range(1, n + 1):
            w[i] = zero
            tempa = _min2(xopt[i] - sl[i], glag[i])
            tempb = _max2(xopt[i] - su[i], glag[i])
            if tempa > zero or tempb < zero:
                w[i] = bigstp
                d1 = glag[i]
                ggfree += d1 * d1
        if ggfree == zero:
            return alpha, zero
        while True:  # L120
            temp = adelt * adelt - wfixsq
            if temp > zero:
                wsqsav = wfixsq
                step = _sqrt(temp / ggfree)
                ggfree = zero
                for i in range(1, n + 1):
                    if w[i] == bigstp:
                        temp = xopt[i] - step * glag[i]
                        if temp <= sl[i]:
                            w[i] = sl[i] - xopt[i]
                            d1 = w[i]
                            wfixsq += d1 * d1
                        elif temp >= su[i]:
                            w[i] = su[i] - xopt[i]
                            d1 = w[i]
                            wfixsq += d1 * d1
                        else:
                            d1 = glag[i]
                            ggfree += d1 * d1
                if wfixsq > wsqsav and ggfree > zero:
                    continue
            break
        # The remaining free components of W and all components of XALT.
        gw = zero
        for i in range(1, n + 1):
            if w[i] == bigstp:
                w[i] = -step * glag[i]
                xalt[i] = _max2(sl[i], _min2(su[i], xopt[i] + w[i]))
            elif w[i] == zero:
                xalt[i] = xopt[i]
            elif glag[i] > zero:
                xalt[i] = sl[i]
            else:
                xalt[i] = su[i]
            gw += glag[i] * w[i]
        # CURV along W; scale W if that reduces |Lagrange function| at XOPT+W.
        curv = zero
        for k in range(1, npt + 1):
            temp = zero
            for j in range(1, n + 1):
                temp += xpt[k + j * xpt_dim1] * w[j]
            curv += hcol[k] * temp * temp
        if iflag == 1:
            curv = -curv
        if curv > -gw and curv < -const * gw:
            scale = -gw / curv
            for i in range(1, n + 1):
                temp = xopt[i] + scale * w[i]
                xalt[i] = _max2(sl[i], _min2(su[i], temp))
            d1 = half * gw * scale
            cauchy = d1 * d1
        else:
            d1 = gw + half * curv
            cauchy = d1 * d1
        # With IFLAG zero, repeat with -GLAG and keep the XALT with the larger CAUCHY.
        if iflag == 0:
            for i in range(1, n + 1):
                glag[i] = -glag[i]
                w[n + i] = xalt[i]
            csave = cauchy
            iflag = 1
            continue
        if csave > cauchy:
            for i in range(1, n + 1):
                xalt[i] = w[n + i]
            cauchy = csave
        return alpha, cauchy


def _trsbox(n, npt, xpt_v, xopt_v, gopt_v, hq_v, pq_v, sl_v, su_v, delta, xnew_v, d_v,
            gnew_v, xbdi_v, s_v, hs_v, hred_v):  # fmt: skip
    """Returns ``(dsq, crvmin)``."""
    xpt_dim1 = npt
    xpt = _P(xpt_v, 1 + xpt_dim1)
    xopt = _P(xopt_v, 1)
    gopt = _P(gopt_v, 1)
    hq = _P(hq_v, 1)
    pq = _P(pq_v, 1)
    sl = _P(sl_v, 1)
    su = _P(su_v, 1)
    xnew = _P(xnew_v, 1)
    d = _P(d_v, 1)
    gnew = _P(gnew_v, 1)
    xbdi = _P(xbdi_v, 1)
    s = _P(s_v, 1)
    hs = _P(hs_v, 1)
    hred = _P(hred_v, 1)

    iact = 0
    xsav = D(0.0)
    angbd = D(0.0)
    dredg = D(0.0)
    sredg = D(0.0)
    ggsav = D(0.0)
    dredsq = D(0.0)
    gredsq = D(0.0)
    itcsav = 0
    rdprev = D(0.0)
    rdnext = D(0.0)
    itermax = 0
    blen = stplen = stepsq = beta = sdec = D(0.0)
    half = D(0.5)
    one = D(1.0)
    onemin = D(-1.0)
    zero = D(0.0)

    # XBDI marks variables fixed at a bound initially; D and GNEW for the first iteration.
    iterc = 0
    nact = 0
    for i in range(1, n + 1):
        xbdi[i] = zero
        if xopt[i] <= sl[i]:
            if gopt[i] >= zero:
                xbdi[i] = onemin
        elif xopt[i] >= su[i]:
            if gopt[i] <= zero:
                xbdi[i] = one
        if xbdi[i] != zero:
            nact += 1
        d[i] = zero
        gnew[i] = gopt[i]
    delsq = delta * delta
    qred = zero
    crvmin = onemin

    label = 20
    while True:
        if label == 20:
            beta = zero
            label = 30
        elif label == 30:
            stepsq = zero
            for i in range(1, n + 1):
                if xbdi[i] != zero:
                    s[i] = zero
                elif beta == zero:
                    s[i] = -gnew[i]
                else:
                    s[i] = beta * s[i] - gnew[i]
                d1 = s[i]
                stepsq += d1 * d1
            if stepsq == zero:
                label = 190
                continue
            if beta == zero:
                gredsq = stepsq
                itermax = iterc + n - nact
            if gredsq * delsq <= qred * D(1e-4) * qred:
                label = 190
                continue
            label = 210
        elif label == 50:
            resid = delsq
            ds = zero
            shs = zero
            for i in range(1, n + 1):
                if xbdi[i] == zero:
                    d1 = d[i]
                    resid -= d1 * d1
                    ds += s[i] * d[i]
                    shs += s[i] * hs[i]
            if resid <= zero:
                label = 90
                continue
            temp = _sqrt(stepsq * resid + ds * ds)
            if ds < zero:
                blen = (temp - ds) / stepsq
            else:
                blen = resid / (temp + ds)
            stplen = blen
            if shs > zero:
                stplen = _min2(blen, gredsq / shs)
            # Reduce STPLEN to preserve the simple bounds; IACT = new constrained variable.
            iact = 0
            for i in range(1, n + 1):
                if s[i] != zero:
                    xsum = xopt[i] + d[i]
                    if s[i] > zero:
                        temp = (su[i] - xsum) / s[i]
                    else:
                        temp = (sl[i] - xsum) / s[i]
                    if temp < stplen:
                        stplen = temp
                        iact = i
            # Update CRVMIN, GNEW and D. SDEC is the decrease in Q.
            sdec = zero
            if stplen > zero:
                iterc += 1
                temp = shs / stepsq
                if iact == 0 and temp > zero:
                    crvmin = _min2(crvmin, temp)
                    if crvmin == onemin:
                        crvmin = temp
                ggsav = gredsq
                gredsq = zero
                for i in range(1, n + 1):
                    gnew[i] += stplen * hs[i]
                    if xbdi[i] == zero:
                        d1 = gnew[i]
                        gredsq += d1 * d1
                    d[i] += stplen * s[i]
                sdec = _max2(stplen * (ggsav - half * stplen * shs), zero)
                qred += sdec
            # Restart the conjugate gradient method if it has hit a new bound.
            if iact > 0:
                nact += 1
                xbdi[iact] = one
                if s[iact] < zero:
                    xbdi[iact] = onemin
                d1 = d[iact]
                delsq -= d1 * d1
                if delsq <= zero:
                    label = 90
                    continue
                label = 20
                continue
            if stplen < blen:
                if iterc == itermax:
                    label = 190
                    continue
                if sdec <= qred * D(0.01):
                    label = 190
                    continue
                beta = gredsq / ggsav
                label = 30
                continue
            label = 90
        elif label == 90:
            crvmin = zero
            label = 100
        elif label == 100:
            if nact >= n - 1:
                label = 190
                continue
            dredsq = zero
            dredg = zero
            gredsq = zero
            for i in range(1, n + 1):
                if xbdi[i] == zero:
                    d1 = d[i]
                    dredsq += d1 * d1
                    dredg += d[i] * gnew[i]
                    d1 = gnew[i]
                    gredsq += d1 * d1
                    s[i] = d[i]
                else:
                    s[i] = zero
            itcsav = iterc
            label = 210
        elif label == 120:
            iterc += 1
            temp = gredsq * dredsq - dredg * dredg
            if temp <= qred * D(1e-4) * qred:
                label = 190
                continue
            temp = _sqrt(temp)
            for i in range(1, n + 1):
                if xbdi[i] == zero:
                    s[i] = (dredg * d[i] - dredsq * gnew[i]) / temp
                else:
                    s[i] = zero
            sredg = -temp
            # Upper bound ANGBD on the tangent of half the angle of the alternative iteration.
            angbd = one
            iact = 0
            jump = False
            for i in range(1, n + 1):
                if xbdi[i] == zero:
                    tempa = xopt[i] + d[i] - sl[i]
                    tempb = su[i] - xopt[i] - d[i]
                    if tempa <= zero:
                        nact += 1
                        xbdi[i] = onemin
                        jump = True
                        break
                    elif tempb <= zero:
                        nact += 1
                        xbdi[i] = one
                        jump = True
                        break
                    d1 = d[i]
                    d2 = s[i]
                    ssq = d1 * d1 + d2 * d2
                    d1 = xopt[i] - sl[i]
                    temp = ssq - d1 * d1
                    if temp > zero:
                        temp = _sqrt(temp) - s[i]
                        if angbd * temp > tempa:
                            angbd = tempa / temp
                            iact = i
                            xsav = onemin
                    d1 = su[i] - xopt[i]
                    temp = ssq - d1 * d1
                    if temp > zero:
                        temp = _sqrt(temp) + s[i]
                        if angbd * temp > tempb:
                            angbd = tempb / temp
                            iact = i
                            xsav = one
            label = 100 if jump else 210
        elif label == 150:
            shs = zero
            dhs = zero
            dhd = zero
            for i in range(1, n + 1):
                if xbdi[i] == zero:
                    shs += s[i] * hs[i]
                    dhs += d[i] * hs[i]
                    dhd += d[i] * hred[i]
            # Greatest reduction in Q over equally spaced ANGT in [0, ANGBD].
            redmax = zero
            isav = 0
            redsav = zero
            iu = int(angbd * D(17.0) + D(3.1))
            for i in range(1, iu + 1):
                angt = angbd * D(i) / D(iu)
                sth = (angt + angt) / (one + angt * angt)
                temp = shs + angt * (angt * dhd - dhs - dhs)
                rednew = sth * (angt * dredg - sredg - half * sth * temp)
                if rednew > redmax:
                    redmax = rednew
                    isav = i
                    rdprev = redsav
                elif i == isav + 1:
                    rdnext = rednew
                redsav = rednew
            if isav == 0:
                label = 190
                continue
            if isav < iu:
                temp = (rdnext - rdprev) / (redmax + redmax - rdprev - rdnext)
                angt = angbd * (D(isav) + half * temp) / D(iu)
            cth = (one - angt * angt) / (one + angt * angt)
            sth = (angt + angt) / (one + angt * angt)
            temp = shs + angt * (angt * dhd - dhs - dhs)
            sdec = sth * (angt * dredg - sredg - half * sth * temp)
            if sdec <= zero:
                label = 190
                continue
            # Update GNEW, D and HRED.
            dredg = zero
            gredsq = zero
            for i in range(1, n + 1):
                gnew[i] = gnew[i] + (cth - one) * hred[i] + sth * hs[i]
                if xbdi[i] == zero:
                    d[i] = cth * d[i] + sth * s[i]
                    dredg += d[i] * gnew[i]
                    d1 = gnew[i]
                    gredsq += d1 * d1
                hred[i] = cth * hred[i] + sth * hs[i]
            qred += sdec
            if iact > 0 and isav == iu:
                nact += 1
                xbdi[iact] = xsav
                label = 100
                continue
            if sdec > qred * D(0.01):
                label = 120
                continue
            label = 190
        elif label == 190:
            dsq = zero
            for i in range(1, n + 1):
                xnew[i] = _max2(_min2(xopt[i] + d[i], su[i]), sl[i])
                if xbdi[i] == onemin:
                    xnew[i] = sl[i]
                if xbdi[i] == one:
                    xnew[i] = su[i]
                d[i] = xnew[i] - xopt[i]
                d1 = d[i]
                dsq += d1 * d1
            return dsq, crvmin
        else:  # label 210: HS = (second derivative matrix of Q) * S
            ih = 0
            for j in range(1, n + 1):
                hs[j] = zero
                for i in range(1, j + 1):
                    ih += 1
                    if i < j:
                        hs[j] += hq[ih] * s[i]
                    hs[i] += hq[ih] * s[j]
            for k in range(1, npt + 1):
                if pq[k] != zero:
                    temp = zero
                    for j in range(1, n + 1):
                        temp += xpt[k + j * xpt_dim1] * s[j]
                    temp *= pq[k]
                    for i in range(1, n + 1):
                        hs[i] += temp * xpt[k + i * xpt_dim1]
            if crvmin != zero:
                label = 50
                continue
            if iterc > itcsav:
                label = 150
                continue
            for i in range(1, n + 1):
                hred[i] = hs[i]
            label = 120


def _prelim(n, npt, x_v, xl_v, xu_v, rhobeg, stop, calfun, xbase_v, xpt_v, fval_v, gopt_v,
            hq_v, pq_v, bmat_v, zmat_v, ndim, sl_v, su_v):  # fmt: skip
    """Returns ``(rc, kopt)``."""
    zmat_dim1 = npt
    zmat = _P(zmat_v, 1 + zmat_dim1)
    xpt_dim1 = npt
    xpt = _P(xpt_v, 1 + xpt_dim1)
    x = _P(x_v, 1)
    xl = _P(xl_v, 1)
    xu = _P(xu_v, 1)
    xbase = _P(xbase_v, 1)
    fval = _P(fval_v, 1)
    gopt = _P(gopt_v, 1)
    hq = _P(hq_v, 1)
    pq = _P(pq_v, 1)
    bmat_dim1 = ndim
    bmat = _P(bmat_v, 1 + bmat_dim1)
    sl = _P(sl_v, 1)
    su = _P(su_v, 1)

    ipt = jpt = 0
    fbeg = D(0.0)
    stepa = D(0.0)
    stepb = D(0.0)
    kopt = 0
    half = D(0.5)
    one = D(1.0)
    two = D(2.0)
    zero = D(0.0)
    rhosq = rhobeg * rhobeg
    recip = one / rhosq
    np_ = n + 1

    # XBASE = initial X; XPT, BMAT, HQ, PQ, ZMAT = 0.
    for j in range(1, n + 1):
        xbase[j] = x[j]
        for k in range(1, npt + 1):
            xpt[k + j * xpt_dim1] = zero
        for i in range(1, ndim + 1):
            bmat[i + j * bmat_dim1] = zero
    for ih in range(1, n * np_ // 2 + 1):
        hq[ih] = zero
    for k in range(1, npt + 1):
        pq[k] = zero
        for j in range(1, npt - np_ + 1):
            zmat[k + j * zmat_dim1] = zero

    nf = 0
    while True:  # L50
        nfm = nf
        nfx = nf - n
        nf += 1
        if nfm <= n << 1:
            if 1 <= nfm <= n:
                stepa = rhobeg
                if su[nfm] == zero:
                    stepa = -stepa
                xpt[nf + nfm * xpt_dim1] = stepa
            elif nfm > n:
                stepa = xpt[nf - n + nfx * xpt_dim1]
                stepb = -rhobeg
                if sl[nfx] == zero:
                    stepb = _min2(two * rhobeg, su[nfx])
                if su[nfx] == zero:
                    stepb = _max2(-two * rhobeg, sl[nfx])
                xpt[nf + nfx * xpt_dim1] = stepb
        else:
            itemp = (nfm - np_) // n
            jpt = nfm - itemp * n - n
            ipt = jpt + itemp
            if ipt > n:
                itemp = jpt
                jpt = ipt - n
                ipt = itemp
            xpt[nf + ipt * xpt_dim1] = xpt[ipt + 1 + ipt * xpt_dim1]
            xpt[nf + jpt * xpt_dim1] = xpt[jpt + 1 + jpt * xpt_dim1]

        # The next value of F; the least value so far and its index.
        for j in range(1, n + 1):
            x[j] = _min2(_max2(xl[j], xbase[j] + xpt[nf + j * xpt_dim1]), xu[j])
            if xpt[nf + j * xpt_dim1] == sl[j]:
                x[j] = xl[j]
            if xpt[nf + j * xpt_dim1] == su[j]:
                x[j] = xu[j]
        stop.nf += 1
        f = D(calfun(x.ptr(1)[:n].copy()))
        fval[nf] = f
        if nf == 1:
            fbeg = f
            kopt = 1
        elif f < fval[kopt]:
            kopt = nf

        # Initial BMAT and quadratic model for NF <= 2N+1.
        if nf <= (n << 1) + 1:
            if 2 <= nf <= n + 1:
                gopt[nfm] = (f - fbeg) / stepa
                if npt < nf + n:
                    bmat[nfm * bmat_dim1 + 1] = -one / stepa
                    bmat[nf + nfm * bmat_dim1] = one / stepa
                    bmat[npt + nfm + nfm * bmat_dim1] = -half * rhosq
            elif nf >= n + 2:
                ih = nfx * (nfx + 1) // 2
                temp = (f - fbeg) / stepb
                diff = stepb - stepa
                hq[ih] = two * (temp - gopt[nfx]) / diff
                gopt[nfx] = (gopt[nfx] * stepb - temp * stepa) / diff
                if stepa * stepb < zero:
                    if f < fval[nf - n]:
                        fval[nf] = fval[nf - n]
                        fval[nf - n] = f
                        if kopt == nf:
                            kopt = nf - n
                        xpt[nf - n + nfx * xpt_dim1] = stepb
                        xpt[nf + nfx * xpt_dim1] = stepa
                bmat[nfx * bmat_dim1 + 1] = -(stepa + stepb) / (stepa * stepb)
                bmat[nf + nfx * bmat_dim1] = -half / xpt[nf - n + nfx * xpt_dim1]
                bmat[nf - n + nfx * bmat_dim1] = (
                    -bmat[nfx * bmat_dim1 + 1] - bmat[nf + nfx * bmat_dim1]
                )
                zmat[nfx * zmat_dim1 + 1] = _sqrt(two) / (stepa * stepb)
                zmat[nf + nfx * zmat_dim1] = _sqrt(half) / rhosq
                zmat[nf - n + nfx * zmat_dim1] = (
                    -zmat[nfx * zmat_dim1 + 1] - zmat[nf + nfx * zmat_dim1]
                )
        else:
            # Off-diagonal second derivatives of the Lagrange functions and the model.
            ih = ipt * (ipt - 1) // 2 + jpt
            zmat[nfx * zmat_dim1 + 1] = recip
            zmat[nf + nfx * zmat_dim1] = recip
            zmat[ipt + 1 + nfx * zmat_dim1] = -recip
            zmat[jpt + 1 + nfx * zmat_dim1] = -recip
            temp = xpt[nf + ipt * xpt_dim1] * xpt[nf + jpt * xpt_dim1]
            hq[ih] = (fbeg - fval[ipt + 1] - fval[jpt + 1] + f) / temp
        if stop.evals():
            return MAXFUN_REACHED, kopt
        if nf < npt:
            continue
        return SUCCESS, kopt


def _bobyqb(n, npt, x_v, xl_v, xu_v, rhobeg, rhoend, stop, calfun, xbase_v, xpt_v, fval_v,
            xopt_v, gopt_v, hq_v, pq_v, bmat_v, zmat_v, ndim, sl_v, su_v, xnew_v, xalt_v, d_v,
            vlag_v, w_v):  # fmt: skip
    """Returns ``(rc, minf)``; ``x_v`` holds the result."""
    zmat_dim1 = npt
    zmat = _P(zmat_v, 1 + zmat_dim1)
    xpt_dim1 = npt
    xpt = _P(xpt_v, 1 + xpt_dim1)
    x = _P(x_v, 1)
    xl = _P(xl_v, 1)
    xu = _P(xu_v, 1)
    xbase = _P(xbase_v, 1)
    fval = _P(fval_v, 1)
    xopt = _P(xopt_v, 1)
    gopt = _P(gopt_v, 1)
    hq = _P(hq_v, 1)
    pq = _P(pq_v, 1)
    bmat_dim1 = ndim
    bmat = _P(bmat_v, 1 + bmat_dim1)
    sl = _P(sl_v, 1)
    su = _P(su_v, 1)
    xnew = _P(xnew_v, 1)
    xalt = _P(xalt_v, 1)
    d = _P(d_v, 1)
    vlag = _P(vlag_v, 1)
    w = _P(w_v, 1)

    diffc = D(0.0)
    alpha = D(0.0)
    adelt = D(0.0)
    ratio = D(0.0)
    cauchy = D(0.0)
    half = D(0.5)
    one = D(1.0)
    ten = D(10.0)
    tenth = D(0.1)
    two = D(2.0)
    zero = D(0.0)
    np_ = n + 1
    nptm = npt - np_
    nh = n * np_ // 2
    rc = SUCCESS
    knew = 0
    dsq = crvmin = dnorm = distsq = denom = beta = vquad = fopt = diff = D(0.0)
    f = D(0.0)

    # PRELIM sets XBASE, XPT, FVAL, GOPT, HQ, PQ, BMAT, ZMAT, NF and KOPT; then the initial XOPT.
    rc2, kopt = _prelim(n, npt, x.ptr(1), xl.ptr(1), xu.ptr(1), rhobeg, stop, calfun,
                        xbase.ptr(1), xpt.ptr(1 + xpt_dim1), fval.ptr(1), gopt.ptr(1), hq.ptr(1),
                        pq.ptr(1), bmat.ptr(1 + bmat_dim1), zmat.ptr(1 + zmat_dim1), ndim,
                        sl.ptr(1), su.ptr(1))  # fmt: skip
    xoptsq = zero
    for i in range(1, n + 1):
        xopt[i] = xpt[kopt + i * xpt_dim1]
        d1 = xopt[i]
        xoptsq += d1 * d1
    fsave = fval[1]
    if rc2 != SUCCESS:
        rc = rc2
        label = 720
    else:
        kbase = 1
        rho = rhobeg
        delta = rho
        nresc = stop.nf
        ntrits = 0
        diffa = zero
        diffb = zero
        itest = 0
        nfsav = stop.nf
        label = 20

    while True:
        if label == 20:
            # Update GOPT before the first iteration and after RESCUE calls CALFUN.
            if kopt != kbase:
                ih = 0
                for j in range(1, n + 1):
                    for i in range(1, j + 1):
                        ih += 1
                        if i < j:
                            gopt[j] += hq[ih] * xopt[i]
                        gopt[i] += hq[ih] * xopt[j]
                if stop.nf > npt:
                    for k in range(1, npt + 1):
                        temp = zero
                        for j in range(1, n + 1):
                            temp += xpt[k + j * xpt_dim1] * xopt[j]
                        temp = pq[k] * temp
                        for i in range(1, n + 1):
                            gopt[i] += temp * xpt[k + i * xpt_dim1]
            label = 60
        elif label == 60:
            # Next point in the trust region (TRSBOX); NTRITS = -1 if ||XNEW-XOPT|| < RHO/2.
            dsq, crvmin = _trsbox(n, npt, xpt.ptr(1 + xpt_dim1), xopt.ptr(1), gopt.ptr(1),
                                  hq.ptr(1), pq.ptr(1), sl.ptr(1), su.ptr(1), delta, xnew.ptr(1),
                                  d.ptr(1), w.ptr(1), w.ptr(np_), w.ptr(np_ + n),
                                  w.ptr(np_ + (n << 1)), w.ptr(np_ + n * 3))  # fmt: skip
            dnorm = _min2(delta, _sqrt(dsq))
            if dnorm < half * rho:
                ntrits = -1
                d1 = ten * rho
                distsq = d1 * d1
                if stop.nf <= nfsav + 2:
                    label = 650
                    continue
                errbig = _max2(_max2(diffa, diffb), diffc)
                frhosq = rho * D(0.125) * rho
                if crvmin > zero and errbig > frhosq * crvmin:
                    label = 650
                    continue
                bdtol = errbig / rho
                go650 = False
                for j in range(1, n + 1):
                    bdtest = bdtol
                    if xnew[j] == sl[j]:
                        bdtest = w[j]
                    if xnew[j] == su[j]:
                        bdtest = -w[j]
                    if bdtest < bdtol:
                        curv = hq[(j + j * j) // 2]
                        for k in range(1, npt + 1):
                            d1 = xpt[k + j * xpt_dim1]
                            curv += pq[k] * (d1 * d1)
                        bdtest += half * curv * rho
                        if bdtest < bdtol:
                            go650 = True
                            break
                label = 650 if go650 else 680
                continue
            ntrits += 1
            label = 90
        elif label == 90:
            # Shift XBASE if XOPT is too far from it (severe cancellation otherwise).
            if dsq <= xoptsq * D(0.001):
                fracsq = xoptsq * D(0.25)
                sumpq = zero
                for k in range(1, npt + 1):
                    sumpq += pq[k]
                    sum_ = -half * xoptsq
                    for i in range(1, n + 1):
                        sum_ += xpt[k + i * xpt_dim1] * xopt[i]
                    w[npt + k] = sum_
                    temp = fracsq - half * sum_
                    for i in range(1, n + 1):
                        w[i] = bmat[k + i * bmat_dim1]
                        vlag[i] = sum_ * xpt[k + i * xpt_dim1] + temp * xopt[i]
                        ip = npt + i
                        for j in range(1, i + 1):
                            bmat[ip + j * bmat_dim1] = (
                                bmat[ip + j * bmat_dim1] + w[i] * vlag[j] + vlag[i] * w[j]
                            )
                # Revisions of BMAT that depend on ZMAT.
                for jj in range(1, nptm + 1):
                    sumz = zero
                    sumw = zero
                    for k in range(1, npt + 1):
                        sumz += zmat[k + jj * zmat_dim1]
                        vlag[k] = w[npt + k] * zmat[k + jj * zmat_dim1]
                        sumw += vlag[k]
                    for j in range(1, n + 1):
                        sum_ = (fracsq * sumz - half * sumw) * xopt[j]
                        for k in range(1, npt + 1):
                            sum_ += vlag[k] * xpt[k + j * xpt_dim1]
                        w[j] = sum_
                        for k in range(1, npt + 1):
                            bmat[k + j * bmat_dim1] += sum_ * zmat[k + jj * zmat_dim1]
                    for i in range(1, n + 1):
                        ip = i + npt
                        temp = w[i]
                        for j in range(1, i + 1):
                            bmat[ip + j * bmat_dim1] += temp * w[j]
                # Complete the shift, including the second derivative parameters of the model.
                ih = 0
                for j in range(1, n + 1):
                    w[j] = -half * sumpq * xopt[j]
                    for k in range(1, npt + 1):
                        w[j] += pq[k] * xpt[k + j * xpt_dim1]
                        xpt[k + j * xpt_dim1] -= xopt[j]
                    for i in range(1, j + 1):
                        ih += 1
                        hq[ih] = hq[ih] + w[i] * xopt[j] + xopt[i] * w[j]
                        bmat[npt + i + j * bmat_dim1] = bmat[npt + j + i * bmat_dim1]
                for i in range(1, n + 1):
                    xbase[i] += xopt[i]
                    xnew[i] -= xopt[i]
                    sl[i] -= xopt[i]
                    su[i] -= xopt[i]
                    xopt[i] = zero
                xoptsq = zero
            label = 210 if ntrits == 0 else 230
        elif label == 190:
            # RESCUE: new BMAT and ZMAT from scratch (rounding errors damaged the denominator).
            nfsav = stop.nf
            kbase = kopt
            rc2, kopt = _rescue(n, npt, xl.ptr(1), xu.ptr(1), stop, calfun, xbase.ptr(1),
                                xpt.ptr(1 + xpt_dim1), fval.ptr(1), xopt.ptr(1), gopt.ptr(1),
                                hq.ptr(1), pq.ptr(1), bmat.ptr(1 + bmat_dim1),
                                zmat.ptr(1 + zmat_dim1), ndim, sl.ptr(1), su.ptr(1), delta,
                                kopt, vlag.ptr(1), w.ptr(1), w.ptr(n + np_), w.ptr(ndim + np_))  # fmt: skip
            xoptsq = zero
            if kopt != kbase:
                for i in range(1, n + 1):
                    xopt[i] = xpt[kopt + i * xpt_dim1]
                    d1 = xopt[i]
                    xoptsq += d1 * d1
            if rc2 != SUCCESS:
                rc = rc2
                label = 720
                continue
            nresc = stop.nf
            if nfsav < stop.nf:
                nfsav = stop.nf
                label = 20
                continue
            if ntrits > 0:
                label = 60
                continue
            label = 210
        elif label == 210:
            # ALTMOV: two alternative new positions XNEW, XALT of the KNEW-th point.
            alpha, cauchy = _altmov(n, npt, xpt.ptr(1 + xpt_dim1), xopt.ptr(1),
                                    bmat.ptr(1 + bmat_dim1), zmat.ptr(1 + zmat_dim1), ndim,
                                    sl.ptr(1), su.ptr(1), kopt, knew, adelt, xnew.ptr(1),
                                    xalt.ptr(1), w.ptr(1), w.ptr(np_), w.ptr(ndim + 1))  # fmt: skip
            for i in range(1, n + 1):
                d[i] = xnew[i] - xopt[i]
            label = 230
        elif label == 230:
            # VLAG and BETA for the current choice of D; D.XPT(K,.) kept in W(NPT+K).
            for k in range(1, npt + 1):
                suma = zero
                sumb = zero
                sum_ = zero
                for j in range(1, n + 1):
                    suma += xpt[k + j * xpt_dim1] * d[j]
                    sumb += xpt[k + j * xpt_dim1] * xopt[j]
                    sum_ += bmat[k + j * bmat_dim1] * d[j]
                w[k] = suma * (half * suma + sumb)
                vlag[k] = sum_
                w[npt + k] = suma
            beta = zero
            for jj in range(1, nptm + 1):
                sum_ = zero
                for k in range(1, npt + 1):
                    sum_ += zmat[k + jj * zmat_dim1] * w[k]
                beta -= sum_ * sum_
                for k in range(1, npt + 1):
                    vlag[k] += sum_ * zmat[k + jj * zmat_dim1]
            dsq = zero
            bsum = zero
            dx = zero
            for j in range(1, n + 1):
                d1 = d[j]
                dsq += d1 * d1
                sum_ = zero
                for k in range(1, npt + 1):
                    sum_ += w[k] * bmat[k + j * bmat_dim1]
                bsum += sum_ * d[j]
                jp = npt + j
                for i in range(1, n + 1):
                    sum_ += bmat[jp + i * bmat_dim1] * d[i]
                vlag[jp] = sum_
                bsum += sum_ * d[j]
                dx += d[j] * xopt[j]
            beta = dx * dx + dsq * (xoptsq + dx + dx + half * dsq) + beta - bsum
            vlag[kopt] += one

            if ntrits == 0:
                # Maybe replace the ALTMOV step by a Cauchy step; RESCUE if rounding hurt.
                d1 = vlag[knew]
                denom = d1 * d1 + alpha * beta
                if denom < cauchy and cauchy > zero:
                    for i in range(1, n + 1):
                        xnew[i] = xalt[i]
                        d[i] = xnew[i] - xopt[i]
                    cauchy = zero
                    label = 230
                    continue
                d1 = vlag[knew]
                if denom <= half * (d1 * d1):
                    if stop.nf > nresc:
                        label = 190
                        continue
                    rc = ROUNDOFF_LIMITED
                    label = 720
                    continue
            else:
                # KNEW: the point to delete to make room for a trust region step.
                delsq = delta * delta
                scaden = zero
                biglsq = zero
                knew = 0
                for k in range(1, npt + 1):
                    if k == kopt:
                        continue
                    hdiag = zero
                    for jj in range(1, nptm + 1):
                        d1 = zmat[k + jj * zmat_dim1]
                        hdiag += d1 * d1
                    d1 = vlag[k]
                    den = beta * hdiag + d1 * d1
                    distsq = zero
                    for j in range(1, n + 1):
                        d1 = xpt[k + j * xpt_dim1] - xopt[j]
                        distsq += d1 * d1
                    d3 = distsq / delsq
                    temp = _max2(one, d3 * d3)
                    if temp * den > scaden:
                        scaden = temp * den
                        knew = k
                        denom = den
                    d3 = vlag[k]
                    biglsq = _max2(biglsq, temp * (d3 * d3))
                if scaden <= half * biglsq:
                    if stop.nf > nresc:
                        label = 190
                        continue
                    rc = ROUNDOFF_LIMITED
                    label = 720
                    continue
            label = 360
        elif label == 360:
            # Evaluate F at XBASE+XNEW unless MAXFUN has been reached.
            for i in range(1, n + 1):
                x[i] = _min2(_max2(xl[i], xbase[i] + xnew[i]), xu[i])
                if xnew[i] == sl[i]:
                    x[i] = xl[i]
                if xnew[i] == su[i]:
                    x[i] = xu[i]
            if stop.evals():
                rc = MAXFUN_REACHED
                label = 720
                continue
            stop.nf += 1
            f = D(calfun(x.ptr(1)[:n].copy()))
            if ntrits == -1:
                fsave = f
                rc = RHOEND_REACHED
                if fsave < fval[kopt]:
                    return rc, f
                label = 720
                continue

            # The model's predicted change in F due to D; DIFF is the prediction error.
            fopt = fval[kopt]
            vquad = zero
            ih = 0
            for j in range(1, n + 1):
                vquad += d[j] * gopt[j]
                for i in range(1, j + 1):
                    ih += 1
                    temp = d[i] * d[j]
                    if i == j:
                        temp = half * temp
                    vquad += hq[ih] * temp
            for k in range(1, npt + 1):
                d1 = w[npt + k]
                vquad += half * pq[k] * (d1 * d1)
            diff = f - fopt - vquad
            diffc = diffb
            diffb = diffa
            diffa = abs(diff)
            if dnorm > rho:
                nfsav = stop.nf

            # The next DELTA after a trust region step.
            if ntrits > 0:
                if vquad >= zero:
                    rc = ROUNDOFF_LIMITED
                    label = 720
                    continue
                ratio = (f - fopt) / vquad
                if ratio <= tenth:
                    delta = _min2(half * delta, dnorm)
                elif ratio <= D(0.7):
                    delta = _max2(half * delta, dnorm)
                else:
                    delta = _max2(half * delta, dnorm + dnorm)
                if delta <= rho * D(1.5):
                    delta = rho
                # Recalculate KNEW and DENOM if the new F is less than FOPT.
                if f < fopt:
                    ksav = knew
                    densav = denom
                    delsq = delta * delta
                    scaden = zero
                    biglsq = zero
                    knew = 0
                    for k in range(1, npt + 1):
                        hdiag = zero
                        for jj in range(1, nptm + 1):
                            d1 = zmat[k + jj * zmat_dim1]
                            hdiag += d1 * d1
                        d1 = vlag[k]
                        den = beta * hdiag + d1 * d1
                        distsq = zero
                        for j in range(1, n + 1):
                            d1 = xpt[k + j * xpt_dim1] - xnew[j]
                            distsq += d1 * d1
                        d3 = distsq / delsq
                        temp = _max2(one, d3 * d3)
                        if temp * den > scaden:
                            scaden = temp * den
                            knew = k
                            denom = den
                        d3 = vlag[k]
                        biglsq = _max2(biglsq, temp * (d3 * d3))
                    if scaden <= half * biglsq:
                        knew = ksav
                        denom = densav

            # Update BMAT, ZMAT and the second derivative terms of the model.
            _update(n, npt, bmat.ptr(1 + bmat_dim1), zmat.ptr(1 + zmat_dim1), ndim, vlag.ptr(1),
                    beta, denom, knew, w.ptr(1))  # fmt: skip
            ih = 0
            pqold = pq[knew]
            pq[knew] = zero
            for i in range(1, n + 1):
                temp = pqold * xpt[knew + i * xpt_dim1]
                for j in range(1, i + 1):
                    ih += 1
                    hq[ih] += temp * xpt[knew + j * xpt_dim1]
            for jj in range(1, nptm + 1):
                temp = diff * zmat[knew + jj * zmat_dim1]
                for k in range(1, npt + 1):
                    pq[k] += temp * zmat[k + jj * zmat_dim1]

            # Include the new point; changes to GOPT at the old XOPT.
            fval[knew] = f
            for i in range(1, n + 1):
                xpt[knew + i * xpt_dim1] = xnew[i]
                w[i] = bmat[knew + i * bmat_dim1]
            for k in range(1, npt + 1):
                suma = zero
                for jj in range(1, nptm + 1):
                    suma += zmat[knew + jj * zmat_dim1] * zmat[k + jj * zmat_dim1]
                sumb = zero
                for j in range(1, n + 1):
                    sumb += xpt[k + j * xpt_dim1] * xopt[j]
                temp = suma * sumb
                for i in range(1, n + 1):
                    w[i] += temp * xpt[k + i * xpt_dim1]
            for i in range(1, n + 1):
                gopt[i] += diff * w[i]

            # Update XOPT, GOPT and KOPT if the new F is less than FOPT.
            if f < fopt:
                kopt = knew
                xoptsq = zero
                ih = 0
                for j in range(1, n + 1):
                    xopt[j] = xnew[j]
                    d1 = xopt[j]
                    xoptsq += d1 * d1
                    for i in range(1, j + 1):
                        ih += 1
                        if i < j:
                            gopt[j] += hq[ih] * d[i]
                        gopt[i] += hq[ih] * d[j]
                for k in range(1, npt + 1):
                    temp = zero
                    for j in range(1, n + 1):
                        temp += xpt[k + j * xpt_dim1] * d[j]
                    temp = pq[k] * temp
                    for i in range(1, n + 1):
                        gopt[i] += temp * xpt[k + i * xpt_dim1]

            # Least Frobenius norm interpolant; its gradient at XOPT goes to VLAG(NPT+I).
            if ntrits > 0:
                for k in range(1, npt + 1):
                    vlag[k] = fval[k] - fval[kopt]
                    w[k] = zero
                for j in range(1, nptm + 1):
                    sum_ = zero
                    for k in range(1, npt + 1):
                        sum_ += zmat[k + j * zmat_dim1] * vlag[k]
                    for k in range(1, npt + 1):
                        w[k] += sum_ * zmat[k + j * zmat_dim1]
                for k in range(1, npt + 1):
                    sum_ = zero
                    for j in range(1, n + 1):
                        sum_ += xpt[k + j * xpt_dim1] * xopt[j]
                    w[k + npt] = w[k]
                    w[k] = sum_ * w[k]
                gqsq = zero
                gisq = zero
                for i in range(1, n + 1):
                    sum_ = zero
                    for k in range(1, npt + 1):
                        sum_ = (
                            sum_ + bmat[k + i * bmat_dim1] * vlag[k] + xpt[k + i * xpt_dim1] * w[k]
                        )
                    if xopt[i] == sl[i]:
                        d1 = _min2(zero, gopt[i])
                        gqsq += d1 * d1
                        d1 = _min2(zero, sum_)
                        gisq += d1 * d1
                    elif xopt[i] == su[i]:
                        d1 = _max2(zero, gopt[i])
                        gqsq += d1 * d1
                        d1 = _max2(zero, sum_)
                        gisq += d1 * d1
                    else:
                        d1 = gopt[i]
                        gqsq += d1 * d1
                        gisq += sum_ * sum_
                    vlag[npt + i] = sum_
                # Replace the model by the least Frobenius norm interpolant if the test holds.
                itest += 1
                if gqsq < ten * gisq:
                    itest = 0
                if itest >= 3:
                    for i in range(1, _max2(npt, nh) + 1):
                        if i <= n:
                            gopt[i] = vlag[npt + i]
                        if i <= npt:
                            pq[i] = w[npt + i]
                        if i <= nh:
                            hq[i] = zero
                        itest = 0

            # Another trust region step after a sufficient decrease (or an alternative step).
            if ntrits == 0:
                label = 60
                continue
            if f <= fopt + tenth * vquad:
                label = 60
                continue
            # Are the interpolation points close enough to the best point so far?
            d3 = two * delta
            d4 = ten * rho
            distsq = _max2(d3 * d3, d4 * d4)
            label = 650
        elif label == 650:
            knew = 0
            for k in range(1, npt + 1):
                sum_ = zero
                for j in range(1, n + 1):
                    d1 = xpt[k + j * xpt_dim1] - xopt[j]
                    sum_ += d1 * d1
                if sum_ > distsq:
                    knew = k
                    distsq = sum_
            # KNEW > 0: ALTMOV moves the KNEW-th point within ADELT of XOPT (via label 90).
            if knew > 0:
                dist = _sqrt(distsq)
                if ntrits == -1:
                    delta = _min2(tenth * delta, half * dist)
                    if delta <= rho * D(1.5):
                        delta = rho
                ntrits = 0
                adelt = _max2(_min2(tenth * dist, delta), rho)
                dsq = adelt * adelt
                label = 90
                continue
            if ntrits == -1:
                label = 680
                continue
            if ratio > zero:
                label = 60
                continue
            if _max2(delta, dnorm) > rho:
                label = 60
                continue
            label = 680
        elif label == 680:
            # The work with the current RHO is complete: next RHO and DELTA.
            if rho > rhoend:
                delta = half * rho
                ratio = rho / rhoend
                if ratio <= D(16.0):
                    rho = rhoend
                elif ratio <= D(250.0):
                    rho = _sqrt(ratio) * rhoend
                else:
                    rho = tenth * rho
                delta = _max2(delta, rho)
                ntrits = 0
                nfsav = stop.nf
                label = 60
                continue
            # Return after another Newton-Raphson step if it is too short to have been tried.
            if ntrits == -1:
                label = 360
                continue
            rc = RHOEND_REACHED
            label = 720
        else:  # label 720: X = XBASE + XOPT (best point), within the bounds
            for i in range(1, n + 1):
                x[i] = _min2(_max2(xl[i], xbase[i] + xopt[i]), xu[i])
                if xopt[i] == sl[i]:
                    x[i] = xl[i]
                if xopt[i] == su[i]:
                    x[i] = xu[i]
            return rc, fval[kopt]


def bobyqa(
    calfun: Callable[[np.ndarray], float],
    x0,
    xl,
    xu,
    rhobeg: float,
    rhoend: float,
    maxfun: int,
    npt: int | None = None,
):
    """Minimise ``calfun`` subject to ``xl <= x <= xu`` (Powell's BOBYQA).

    Returns ``(x, f, rc, nf)``: the best point (float64), its value, a return code
    (``SUCCESS``/``RHOEND_REACHED``, ``MAXFUN_REACHED``, ``ROUNDOFF_LIMITED``, ``INVALID_ARGS``)
    and the number of function evaluations.  ``npt`` defaults to ``2n+1`` (``min_bobyqa``).
    """
    x = np.array(x0, dtype=np.float64, copy=True)
    xl_a = np.array(xl, dtype=np.float64, copy=True)
    xu_a = np.array(xu, dtype=np.float64, copy=True)
    n = x.shape[0]
    if npt is None:
        npt = 2 * n + 1
    rhobeg = D(rhobeg)
    rhoend = D(rhoend)
    stop = _Stop(maxfun)
    np_ = n + 1
    if npt < n + 2 or npt > (n + 2) * np_ // 2:
        return x, D(np.nan), INVALID_ARGS, 0

    # Partition the working space (same layout as the Fortran/C driver).
    ndim = npt + n
    ixb = 1
    ixp = ixb + n
    ifv = ixp + n * npt
    ixo = ifv + npt
    igo = ixo + n
    ihq = igo + n
    ipq = ihq + n * np_ // 2
    ibmat = ipq + npt
    izmat = ibmat + ndim * n
    isl = izmat + npt * (npt - np_)
    isu = isl + n
    ixn = isu + n
    ixa = ixn + n
    id_ = ixa + n
    ivl = id_ + n
    iw = ivl + ndim
    w0 = np.zeros((npt + 5) * (npt + n) + 3 * n * (n + 5) // 2, np.float64)
    w = _P(w0, 1)
    xl_p = _P(xl_a, 1)
    xu_p = _P(xu_a, 1)
    x_p = _P(x, 1)

    # Room between the bounds; move X0 off bounds that are within RHOBEG; set SL and SU.
    zero = D(0.0)
    for j in range(1, n + 1):
        temp = xu_p[j] - xl_p[j]
        if temp < rhobeg + rhobeg:
            return x, D(np.nan), INVALID_ARGS, 0
        jsl = isl + j - 1
        jsu = jsl + n
        w[jsl] = xl_p[j] - x_p[j]
        w[jsu] = xu_p[j] - x_p[j]
        if w[jsl] >= -rhobeg:
            if w[jsl] >= zero:
                x_p[j] = xl_p[j]
                w[jsl] = zero
                w[jsu] = temp
            else:
                x_p[j] = xl_p[j] + rhobeg
                w[jsl] = -rhobeg
                w[jsu] = _max2(xu_p[j] - x_p[j], rhobeg)
        elif w[jsu] <= rhobeg:
            if w[jsu] <= zero:
                x_p[j] = xu_p[j]
                w[jsl] = -temp
                w[jsu] = zero
            else:
                x_p[j] = xu_p[j] - rhobeg
                w[jsl] = _min2(xl_p[j] - x_p[j], -rhobeg)
                w[jsu] = rhobeg

    with np.errstate(all="ignore"):
        rc, f = _bobyqb(n, npt, x, xl_a, xu_a, rhobeg, rhoend, stop, calfun, w.ptr(ixb),
                        w.ptr(ixp), w.ptr(ifv), w.ptr(ixo), w.ptr(igo), w.ptr(ihq), w.ptr(ipq),
                        w.ptr(ibmat), w.ptr(izmat), ndim, w.ptr(isl), w.ptr(isu), w.ptr(ixn),
                        w.ptr(ixa), w.ptr(id_), w.ptr(ivl), w.ptr(iw))  # fmt: skip
    return x, f, rc, stop.nf
