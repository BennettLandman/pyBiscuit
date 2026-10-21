"""Sulcal depth: port of ``SulcalDepth/sulcal_depth.m`` (BISCUIT ``SurfQuant``, 2022-02 source).

Chain (orchestrator spec section 9.2), on the pial surface, 1 mm voxels, 20-voxel zero padding:

1. ``outer_hull``: ``v -= min(v)``, ``vox = ceil(max(v))``, ``VOXELISE`` (shared with
   ``OuterHull``), ``padarray``, ``imfill(.., 26, 'holes')`` -> solid ``volume0``;
   ``double*255``, ``smooth3`` gaussian size 9 sigma 2, ``imclose(strel('ball',15,15))``,
   ``imdilate(strel('ball',1,1))``, ``> 25`` -> ``hull`` (morphology shared with :mod:`.hull`).
2. ``distance``: ``field = ~volume0 & hull``, thickened six times by the 18-connected
   ``bwperim`` shells of ``~field`` and of ``volume0``; seeds = ``bwperim(hull, 18)``;
   ``bwdistgeodesic(field, seeds, 'quasi-euclidean')`` (26-neighbour shortest paths with step
   lengths 1, sqrt 2, sqrt 3 inside ``field``; elsewhere NaN -> Inf).
3. ``interp3`` (linear) of the distance field at the vertices (voxel ``k`` at coordinate
   ``k - 1``, shifted by the padding), ``Inf -> 0``.

Provisional: no MATLAB/Octave oracle (see ``docs/ports/sulcal-depth.md``).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from scipy import ndimage

from ..io import read_vtk
from ..mesh import Mesh
from ._voxelise import voxelise
from .hull import (
    CLOSING_H,
    CLOSING_R,
    DILATE_H,
    DILATE_R,
    SMOOTH_SD,
    THRESHOLD,
    _close_ball,
    _dilate_ball,
    _smooth3_gaussian,
)

__all__ = [
    "sulcal_depth",
    "outer_hull_volumes",
    "bwperim",
    "bwdistgeodesic",
    "interp3_linear",
    "write_depth",
]

PAD = 20
SCALE = 1.0
THICKEN = 6
_C18 = ndimage.generate_binary_structure(3, 2)  # conndef(3, 'minimal') is 6; 18 = |offset|^2 <= 2


# --------------------------------------------------------------------------- #
# Image Processing Toolbox equivalents
# --------------------------------------------------------------------------- #
def bwperim(bw: np.ndarray, conn: int = 18) -> np.ndarray:
    """``bwperim(BW, conn)``: set voxels with at least one unset ``conn``-neighbour.

    Voxels outside the array count as set (the toolbox erodes with ones padding), so the array
    border alone never makes a perimeter.  Unverified against MATLAB (spec section 7).
    """
    if conn != 18:
        raise NotImplementedError("only 18-connectivity is used by sulcal_depth.m")
    bw = np.asarray(bw, dtype=bool)
    return bw & ~ndimage.binary_erosion(bw, structure=_C18, border_value=1)


_OFFS = np.array(
    [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if i or j or k],
    dtype=np.int64,
)
_STEP = np.sqrt((_OFFS**2).sum(axis=1)).astype(np.float64)  # 1, sqrt2, sqrt3


@njit(cache=True)
def _dijkstra(mask, seeds, offs, step):
    nx, ny, nz = mask.shape
    n = nx * ny * nz
    dist = np.full(n, np.inf)
    # binary min-heap of (key, node) with lazy deletion
    cap = 1 << 16
    hk = np.empty(cap)
    hn = np.empty(cap, np.int64)
    size = 0
    flat = mask.ravel()
    for s in seeds:
        dist[s] = 0.0
        if size == cap:
            cap *= 2
            hk2 = np.empty(cap)
            hn2 = np.empty(cap, np.int64)
            hk2[:size] = hk[:size]
            hn2[:size] = hn[:size]
            hk, hn = hk2, hn2
        i = size
        size += 1
        hk[i] = 0.0
        hn[i] = s
        while i > 0 and hk[(i - 1) >> 1] > hk[i]:
            p = (i - 1) >> 1
            hk[p], hk[i] = hk[i], hk[p]
            hn[p], hn[i] = hn[i], hn[p]
            i = p
    while size > 0:
        d0 = hk[0]
        u = hn[0]
        size -= 1
        if size > 0:
            hk[0] = hk[size]
            hn[0] = hn[size]
            i = 0
            while True:
                c1 = 2 * i + 1
                r = c1 + 1
                m = i
                if c1 < size and hk[c1] < hk[m]:
                    m = c1
                if r < size and hk[r] < hk[m]:
                    m = r
                if m == i:
                    break
                hk[m], hk[i] = hk[i], hk[m]
                hn[m], hn[i] = hn[i], hn[m]
                i = m
        if d0 > dist[u]:
            continue
        ux = u // (ny * nz)
        uy = (u // nz) % ny
        uz = u % nz
        for q in range(offs.shape[0]):
            x = ux + offs[q, 0]
            y = uy + offs[q, 1]
            z = uz + offs[q, 2]
            if x < 0 or y < 0 or z < 0 or x >= nx or y >= ny or z >= nz:
                continue
            v = (x * ny + y) * nz + z
            if not flat[v]:
                continue
            nd = d0 + step[q]
            if nd < dist[v]:
                dist[v] = nd
                if size == cap:
                    cap *= 2
                    hk2 = np.empty(cap)
                    hn2 = np.empty(cap, np.int64)
                    hk2[:size] = hk[:size]
                    hn2[:size] = hn[:size]
                    hk, hn = hk2, hn2
                i = size
                size += 1
                hk[i] = nd
                hn[i] = v
                while i > 0 and hk[(i - 1) >> 1] > hk[i]:
                    p = (i - 1) >> 1
                    hk[p], hk[i] = hk[i], hk[p]
                    hn[p], hn[i] = hn[i], hn[p]
                    i = p
    return dist


def bwdistgeodesic(mask: np.ndarray, seed: np.ndarray, *, seeds_outside_mask: str = "drop"):
    """``bwdistgeodesic(mask, find(seed), 'quasi-euclidean')`` for a 3-D mask.

    Shortest path length through ``mask`` on the 26-neighbourhood with steps 1, sqrt 2, sqrt 3
    from the nearest seed; voxels outside ``mask`` or unreachable are NaN.  Seeds outside the mask
    are dropped (``'drop'``, toolbox behaviour assumed) or used as sources at distance 0
    (``'keep'``); which one MATLAB does is an open item (needs an image run).
    """
    mask = np.ascontiguousarray(mask, dtype=bool)
    seed = np.asarray(seed, dtype=bool)
    if seeds_outside_mask == "drop":
        seed = seed & mask
    elif seeds_outside_mask != "keep":
        raise ValueError("seeds_outside_mask must be 'drop' or 'keep'")
    walk = mask | seed
    idx = np.flatnonzero(seed.ravel()).astype(np.int64)
    d = _dijkstra(walk, idx, _OFFS, _STEP).reshape(mask.shape)
    d[~np.isfinite(d)] = np.nan
    d[~mask & ~seed] = np.nan
    return d


def interp3_linear(vol: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Trilinear interpolation at fractional 0-based indices ``xyz`` (N, 3); NaN outside.

    Weighted sum in IEEE arithmetic, so a zero weight on an ``Inf`` corner gives NaN and a
    positive weight gives ``Inf``, as the MATLAB formula would (unverified, spec section 9.2).
    """
    vol = np.asarray(vol, dtype=np.float64)
    p = np.asarray(xyz, dtype=np.float64)
    hi = np.array(vol.shape) - 1.0
    out_of_range = ((p < 0) | (p > hi)).any(axis=1)
    p = np.clip(p, 0, hi)
    i0 = np.minimum(np.floor(p).astype(np.int64), np.maximum(np.array(vol.shape) - 2, 0))
    t = p - i0
    out = np.zeros(p.shape[0])
    with np.errstate(invalid="ignore"):
        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    w = (
                        (t[:, 0] if a else 1 - t[:, 0])
                        * (t[:, 1] if b else 1 - t[:, 1])
                        * (t[:, 2] if c else 1 - t[:, 2])
                    )
                    out = out + w * vol[i0[:, 0] + a, i0[:, 1] + b, i0[:, 2] + c]
    out[out_of_range] = np.nan
    return out


# --------------------------------------------------------------------------- #
# sulcal_depth.m
# --------------------------------------------------------------------------- #
def outer_hull_volumes(vertices, faces, pad: int = PAD, scale: float = SCALE):
    """``outer_hull`` of ``sulcal_depth.m``: returns ``(volume0 bool, hull bool, v)``.

    ``v`` is the shifted, scaled vertex array (min 0 on every axis).
    """
    v = np.asarray(vertices, dtype=np.float64)
    v = (v - v.min(axis=0)) * scale
    vox = np.ceil(v.max(axis=0)).astype(int)
    vol, *_ = voxelise(v, np.asarray(faces), int(vox[0]), int(vox[1]), int(vox[2]))
    vol = np.pad(vol, pad)
    vol = ndimage.binary_fill_holes(vol, structure=np.ones((3, 3, 3), bool))
    volume0 = vol.copy()
    a = vol.astype(np.float64) * 255.0
    a = _smooth3_gaussian(a, 2 * math.ceil(2 * SMOOTH_SD) + 1, SMOOTH_SD)
    a = _close_ball(a, CLOSING_R, CLOSING_H)
    a = _dilate_ball(a, DILATE_R, DILATE_H)
    return volume0, a > THRESHOLD, v


def _field(volume0: np.ndarray, hull: np.ndarray) -> np.ndarray:
    field = ~volume0 & hull
    tissue_perim = bwperim(volume0, 18)
    for _ in range(THICKEN):
        field = field | bwperim(~field, 18) | tissue_perim
    return field


def sulcal_depth(
    surface, *, pad: int = PAD, scale: float = SCALE, seeds_outside_mask: str = "drop"
) -> np.ndarray:
    """Per-vertex sulcal depth (mm) of a pial surface (:class:`Mesh` or VTK path)."""
    mesh = surface if isinstance(surface, Mesh) else read_vtk(surface)
    volume0, hull, v = outer_hull_volumes(mesh.vertices, mesh.faces, pad, scale)
    field = _field(volume0, hull)
    d = bwdistgeodesic(field, bwperim(hull, 18), seeds_outside_mask=seeds_outside_mask)
    d2 = np.where(np.isnan(d), np.inf, d)  # min(Inf, D) drops NaN
    # voxel k (0-based) sits at coordinate k - pad; v is already in that frame
    sd = interp3_linear(d2, v + pad) / scale
    sd[sd == np.inf] = 0.0
    return sd


def write_depth(sd: np.ndarray, path) -> None:
    """``fprintf(fp, '%f\\n', sd)``."""
    with open(path, "w") as fh:
        fh.writelines(f"{x:f}\n" for x in sd)
