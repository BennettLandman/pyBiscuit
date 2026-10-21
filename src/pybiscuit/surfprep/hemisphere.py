"""MaCRUISE hemisphere separation (plan B5.2; spec §4.2, ``hemi_separation.m``).

Splits the whole-brain MaCRUISE inner surface into two closed hemisphere
meshes.  A cutting plane (see :mod:`pybiscuit.surfprep.contour`) removes the
bridge between the hemispheres, the two open boundary loops are closed with
harmonically filled disk patches, and the seam is smoothed.  Port class:
**replacement-tolerant**: the patch triangulation is our own Delaunay and is
not Qhull-identical to MATLAB's, so only vertices with ``mask == 1`` are
comparable vertex-wise.

Deviations from the MATLAB source (also in ``docs/ports/hemisphere.md``):

* D1 island removal keeps the two *largest* components, not ``dmperm`` blocks 1, 2.
* D2 patch boundary points sit at angles ``2*pi*i/m`` (``m`` distinct points)
  and loop vertex ``i`` is paired with boundary point ``i``; MATLAB pairs in
  Qhull hull order with a duplicated end point.
* D3 on a failed cut (not exactly two loops) the next-ranked plane is tried;
  MATLAB prints "the cutting plane is wrong!" and exits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay

from pybiscuit.mesh.model import Mesh
from pybiscuit.surfprep.contour import (
    DEFAULT_RULE,
    candidate_offsets,
    mesh_contour,
    smooth_neighbor_mean,
)

GRID_N = 50  # circle_patch.m nPts
SEAM_SMOOTH_ITERS = 5
LH_NAME = "lh.target_image_GMimg_innerSurf.vtk"
RH_NAME = "rh.target_image_GMimg_innerSurf.vtk"


class HemisphereError(RuntimeError):
    """The cutting plane did not yield exactly two closed boundary loops."""


# --------------------------------------------------------------------- helpers
def _edges(n, faces):
    f = np.asarray(faces, dtype=np.int64)
    pairs = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    pairs = np.sort(pairs, axis=1)
    return np.unique(pairs, axis=0)


def _adjacency(n, faces, drop=None) -> sp.csr_matrix:
    """Binary symmetric adjacency; edges touching ``drop`` (bool mask) removed."""
    e = _edges(n, faces)
    if drop is not None:
        keep = ~(drop[e[:, 0]] | drop[e[:, 1]])
        e = e[keep]
    r = np.concatenate([e[:, 0], e[:, 1]])
    c = np.concatenate([e[:, 1], e[:, 0]])
    return sp.csr_matrix((np.ones(r.size), (r, c)), shape=(n, n))


def orientation_fix(vertices, faces) -> tuple[np.ndarray, bool]:
    """``surfNormalCorrection``: make normals point away from the +x/-x extreme.

    Returns ``(faces, inverted)``.
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    i0 = int(np.argmin(v[:, 0]))
    fid = np.flatnonzero((f == i0).any(axis=1))
    u1 = v[f[fid, 1]] - v[f[fid, 0]]
    u2 = v[f[fid, 2]] - v[f[fid, 0]]
    u1 /= np.linalg.norm(u1, axis=1)[:, None]
    u2 /= np.linalg.norm(u2, axis=1)[:, None]
    nrm = np.cross(u1, u2)
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    n = nrm.mean(axis=0)
    n /= np.linalg.norm(n)
    nx = np.sign(n[0])
    proj = nx * (v[:, 0] - v[i0, 0])
    if (proj > 0).sum() > (proj <= 0).sum():
        return f[:, ::-1].copy(), True
    return f.copy(), False


def _face_index(n, faces, bdry_mask):
    """``computeIndex``: removed faces, kept faces per vertex, removed count per vertex."""
    f = np.asarray(faces, dtype=np.int64)
    invalid = bdry_mask[f].all(axis=1)
    kept_ids = np.flatnonzero(~invalid)
    v = f[kept_ids].ravel()
    fi = np.repeat(kept_ids, 3)
    order = np.lexsort((fi, v))  # by vertex, then face id (stored order)
    v, fi = v[order], fi[order]
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(ptr, v + 1, 1)
    ptr = np.cumsum(ptr)
    total = np.bincount(f.ravel(), minlength=n)
    kept_cnt = ptr[1:] - ptr[:-1]
    return np.flatnonzero(invalid), (ptr, fi), total - kept_cnt


def _kept_faces(idx, v):
    ptr, fi = idx
    return fi[ptr[v] : ptr[v + 1]]


def _boundary_loops(f, idx, removed_cnt, n):
    """``getBoundary``: two loops by walking kept faces in stored order."""
    ptr, _ = idx
    kept_cnt = ptr[1:] - ptr[:-1]
    pid = np.flatnonzero((kept_cnt > 0) & (removed_cnt > 0))
    if pid.size == 0:
        raise HemisphereError("cutting plane removed no faces")
    in_pid = np.zeros(n, dtype=bool)
    in_pid[pid] = True
    visited = np.zeros(n, dtype=bool)

    def walk(start, tag):
        src = start
        loop = []
        while not visited[start]:
            nxt = -1
            for fc in _kept_faces(idx, src):
                row = f[fc]
                p = int(np.flatnonzero(row == src)[0])
                nv = int(row[(p + 1) % 3])
                if not visited[nv] and in_pid[nv]:
                    nxt = nv
                    break
            if nxt < 0:
                raise HemisphereError(f"the cutting plane is wrong! ({tag}) - {src}")
            visited[nxt] = True
            src = nxt
            loop.append(src)
        return loop[::-1]

    tr1 = walk(int(pid[0]), 1)
    left = np.flatnonzero(~visited[pid])
    if left.size == 0:
        raise HemisphereError("only one boundary loop")
    tr2 = walk(int(pid[left[-1]]), 2)
    if (~visited[pid]).any():
        raise HemisphereError("more than two boundary loops")
    return np.asarray(tr1, dtype=np.int64), np.asarray(tr2, dtype=np.int64)


def _cleanup_boundary(n, f, tr, bdry_mask):
    """``cleanupBoundary``: drop every component except the two largest (D1)."""
    drop = np.zeros(n, dtype=bool)
    drop[tr] = True
    w = _adjacency(n, f, drop)
    used = np.zeros(n, dtype=bool)
    used[np.asarray(f).ravel()] = True
    _, lab = connected_components(w, directed=False)
    sizes = np.bincount(lab[used], minlength=lab.max() + 1)
    top = np.argsort(-sizes, kind="stable")[:2]
    return used & ~np.isin(lab, top) & ~drop


@lru_cache(maxsize=64)
def disk_patch(m: int):
    """Disk triangulation with ``m`` boundary points (``circle_patch.m``, deviation D2).

    Returns ``(points (P,2), triangles (T,3) ccw, boundary_ids (m,))``.  Interior
    points are the 51x51 concentric-disk grid of radius ``1 - 1/50``; boundary
    point ``i`` lies at angle ``2*pi*i/m`` on the unit circle.
    """
    r = 1 - 1 / GRID_N
    scale = min(1.0, 0.9 * np.cos(np.pi / m) / r) if m >= 3 else 1.0
    r1 = np.arange(GRID_N + 1) * (r / GRID_N)
    a = np.repeat(2 * r1 - r, GRID_N + 1)
    b = np.tile(2 * r1 - r, GRID_N + 1)
    upper = a * a > b * b
    radius = np.full(a.shape, r)
    phi = np.zeros(a.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        radius[upper] *= a[upper]
        phi[upper] = (b[upper] / a[upper]) * (np.pi / 4)
        lo = ~upper
        radius[lo] *= b[lo]
        phi[lo] = np.pi / 2 - (np.pi / 4) * (a[lo] / b[lo])
    phi[np.isnan(phi)] = 0
    grid = np.column_stack([np.cos(phi), np.sin(phi)]) * (radius * scale)[:, None]
    ang = 2 * np.pi * np.arange(m) / m
    pts = np.vstack([grid, np.column_stack([np.cos(ang), np.sin(ang)])])
    tri = Delaunay(pts).simplices.astype(np.int64)
    p = pts[tri]
    area = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (
        p[:, 1, 1] - p[:, 0, 1]
    )
    tri[area < 0] = tri[area < 0][:, ::-1]
    return pts, tri, np.arange(grid.shape[0], pts.shape[0])


def harmonic_fill(vertices, loop) -> tuple[np.ndarray, np.ndarray]:
    """``poisson``: unweighted-Laplacian patch with the loop as Dirichlet boundary.

    Returns ``(patch_vertices (P,3), patch_faces (T,3))``; the last ``len(loop)``
    patch vertices coincide with ``vertices[loop]``.
    """
    pts, tri, bid = disk_patch(len(loop))
    n = pts.shape[0]
    e = _edges(n, tri)
    r = np.concatenate([e[:, 0], e[:, 1]])
    c = np.concatenate([e[:, 1], e[:, 0]])
    w = sp.csr_matrix((np.ones(r.size), (r, c)), shape=(n, n))
    a = (sp.diags(np.asarray(w.sum(axis=1)).ravel()) - w).tolil()
    b = np.zeros((n, 3))
    b[bid] = vertices[loop]
    ident = sp.identity(n, format="lil")
    for k in bid:
        a[k] = ident[k]
    x = spla.splu(a.tocsc()).solve(b)
    return x, tri


@dataclass
class SplitDiagnostics:
    plane_index: int = -1
    plane_offset: float = float("nan")
    plane_rank: int = 0
    plane_rule: str = DEFAULT_RULE
    n_cut_faces: int = 0
    n_removed_faces: int = 0
    loop_sizes: tuple[int, int] = (0, 0)
    n_island_vertices: int = 0
    n_patch: dict = field(default_factory=dict)
    orientation_inverted: bool = False
    tried: list = field(default_factory=list)


@dataclass
class HemisphereResult:
    lh: Mesh
    rh: Mesh
    lh_mask: np.ndarray  # int8, 1 original / 0 patch
    rh_mask: np.ndarray
    hemi_map: np.ndarray  # (n_kept, 3) int64: whole-brain id, hemisphere id, 0 lh|1 rh
    diagnostics: SplitDiagnostics


def _split_with_cut(v, f, cut_faces):
    """Everything after the plane choice (steps 3-10) for one candidate cut."""
    n = v.shape[0]
    bd = np.zeros(n, dtype=bool)
    bd[f[cut_faces].ravel()] = True
    # a non-flagged vertex whose neighbours are all flagged joins the boundary
    w = _adjacency(n, f)
    deg = np.asarray(w.sum(axis=1)).ravel()
    nb = w @ bd.astype(float)
    bd |= (~bd) & (deg == nb)
    invalid_f, idx, removed = _face_index(n, f, bd)
    tr1, tr2 = _boundary_loops(f, idx, removed, n)
    tr = np.concatenate([tr1, tr2])
    isl = _cleanup_boundary(n, f, tr, bd)
    n_isl = int(isl.sum())
    if (isl & ~bd).any():
        bd = bd | isl
        invalid_f, idx, removed = _face_index(n, f, bd)

    # patches (circlePatch + poisson) and stitching (closeSurface)
    patches, pf_all, x_all = [], [], []
    for loop in (tr1, tr2):
        x, pf = harmonic_fill(v, loop)
        patches.append(x.shape[0])
        pf_all.append(pf)
        x_all.append(x)
    pv = np.vstack(x_all)
    off = 0
    new_faces, drop_patch = [], []
    for loop, pf, x in zip((tr1, tr2), pf_all, x_all, strict=True):
        bnd = np.arange(x.shape[0] - len(loop), x.shape[0]) + (n + off)
        lut = np.arange(n + pv.shape[0])
        lut[bnd] = loop  # patch boundary points are replaced by the loop vertices
        new_faces.append(lut[pf + (n + off)])
        drop_patch.append(bnd)
        off += x.shape[0]
    v0 = np.vstack([v, pv])
    vorg0 = np.concatenate([np.arange(1, n + 1), np.zeros(pv.shape[0], dtype=np.int64)])
    kept_cnt = idx[0][1:] - idx[0][:-1]
    invalid_v = np.concatenate([kept_cnt == 0, np.zeros(pv.shape[0], dtype=bool)])
    invalid_v[np.concatenate(drop_patch)] = True
    newid = np.cumsum(~invalid_v) - 1
    keep_f = np.ones(f.shape[0], dtype=bool)
    keep_f[invalid_f] = False
    fnew = np.vstack([f[keep_f], *new_faces])
    fnew = newid[fnew]
    vnew = v0[~invalid_v]
    vorgnew = vorg0[~invalid_v]
    m_new = vnew.shape[0]

    # seam smoothing
    w = _adjacency(m_new, fnew)
    seam = np.zeros(m_new, dtype=bool)
    seam[newid[tr]] = True
    for pf in new_faces:
        seam[np.unique(newid[pf])] = True
    rows = np.flatnonzero(seam)
    nbrs = np.unique(w[rows].indices)
    region = np.zeros(m_new, dtype=bool)
    region[nbrs] = True
    vnew0 = smooth_neighbor_mean(vnew, fnew, SEAM_SMOOTH_ITERS, frozen=~region)

    # components -> hemispheres (connectedComp)
    ncomp, lab = connected_components(_adjacency(m_new, fnew), directed=False)
    sizes = np.bincount(lab, minlength=ncomp)
    order = np.lexsort((np.arange(ncomp), -sizes))
    if ncomp < 2:
        raise HemisphereError("the cut left a single connected component")
    c1, c2 = int(order[0]), int(order[1])
    m1 = vnew0[lab == c1].mean(axis=0)
    m2 = vnew0[lab == c2].mean(axis=0)
    if m1[0] > m2[0]:
        c1, c2 = c2, c1
    diag = SplitDiagnostics(
        n_removed_faces=int(invalid_f.size),
        loop_sizes=(len(tr1), len(tr2)),
        n_island_vertices=n_isl,
        n_patch={"loop1": patches[0], "loop2": patches[1]},
    )
    out = []
    bins = np.where(lab == c1, 1, np.where(lab == c2, 2, 3))
    tab = np.zeros(m_new, dtype=np.int64)
    for c in (c1, c2):
        ind = np.flatnonzero(lab == c)
        tab[ind] = np.arange(1, ind.size + 1)
        sel = (lab[fnew] == c).all(axis=1)
        ff = tab[fnew[sel]] - 1
        mask = (vorgnew[ind] > 0).astype(np.int8)
        out.append((vnew0[ind], ff, mask, ind))
    orig = vorgnew > 0
    orig &= bins <= 2
    hemi_map = np.column_stack([vorgnew[orig] - 1, tab[orig] - 1, bins[orig] - 1])
    return out, hemi_map, diag


def separate_hemispheres(
    mesh: Mesh,
    reference=None,
    *,
    rule: str = DEFAULT_RULE,
    max_plane_tries: int = 5,
    fallback: bool = True,
) -> HemisphereResult:
    """Port of ``hemi_separation``.

    ``mesh`` is the native whole-brain inner surface (RAS+, right = +x: the
    smaller-x hemisphere is called ``lh``).  ``reference`` is an ``(N, 3)``
    array of the *same vertices in an alignment frame* whose x axis is
    perpendicular to the mid-sagittal plane (MNI copy, ``aff.txt``-mapped copy,
    or a label-fitted frame; see :mod:`pybiscuit.surfprep.frames`).  ``None``
    uses the native vertices.
    """
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f0 = np.asarray(mesh.faces, dtype=np.int64)
    ref = v if reference is None else np.asarray(reference, dtype=np.float64)
    if ref.shape != v.shape:
        raise ValueError("reference must have the same shape as the mesh vertices")
    f, inverted = orientation_fix(v, f0)
    # getCuttingPlane: orientation fix + 20 smoothing iterations on the reference
    rf, _ = orientation_fix(ref, f0)
    sm = np.round(smooth_neighbor_mean(ref, rf, 20), 6)  # MeshContour reads a 6-decimal VTK
    offsets = candidate_offsets(sm[:, 0])
    ranking, cut = mesh_contour(sm, rf, np.array([1.0, 0.0, 0.0]), offsets, rule)
    tried = []
    last = None
    for rank, pi in enumerate(ranking[: max(1, max_plane_tries if fallback else 1)]):
        try:
            out, hemi_map, diag = _split_with_cut(v, f, cut[pi])
        except HemisphereError as exc:
            tried.append((int(pi), str(exc)))
            last = exc
            continue
        diag.plane_index = int(pi)
        diag.plane_offset = float(offsets[pi])
        diag.plane_rank = rank
        diag.plane_rule = rule
        diag.n_cut_faces = int(cut[pi].size)
        diag.orientation_inverted = inverted
        diag.tried = tried
        (v1, f1, m1, _), (v2, f2, m2, _) = out
        return HemisphereResult(
            lh=Mesh(v1, f1, frame=mesh.frame, units=mesh.units),
            rh=Mesh(v2, f2, frame=mesh.frame, units=mesh.units),
            lh_mask=m1,
            rh_mask=m2,
            hemi_map=hemi_map,
            diagnostics=diag,
        )
    raise HemisphereError(f"no candidate plane gave two boundary loops: {tried}") from last


# --------------------------------------------------------------------- outputs
def _write_vtk_matlab(path, v, f) -> None:
    """``write_vtk.m``: ASCII POLYDATA, ``float`` points with 6 decimals."""
    lines = [
        "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n",
        f"POINTS {v.shape[0]} float\n",
    ]
    body = "\n".join("{:f} {:f} {:f}".format(*r) for r in v.tolist()) + "\n"
    cells = "\n".join("3 {:d} {:d} {:d}".format(*r) for r in f.tolist()) + "\n"
    lines += [body, f"POLYGONS {f.shape[0]} {4 * f.shape[0]}\n", cells]
    Path(path).write_text("".join(lines))


def write_outputs(result: HemisphereResult, out_dir) -> dict[str, Path]:
    """Write ``<h>.target_image_GMimg_innerSurf.vtk``, ``<h>.mask.txt``, ``hemi_map.txt``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for h, mesh, mask in (("lh", result.lh, result.lh_mask), ("rh", result.rh, result.rh_mask)):
        p = out / f"{h}.target_image_GMimg_innerSurf.vtk"
        _write_vtk_matlab(p, mesh.vertices, mesh.faces)
        paths[f"{h}_surf"] = p
        pm = out / f"{h}.mask.txt"
        pm.write_text("".join(f"{int(x)}\n" for x in mask))
        paths[f"{h}_mask"] = pm
    pt = out / "hemi_map.txt"
    pt.write_text("".join("{:d} {:d} {:d}\n".format(*r) for r in result.hemi_map.tolist()))
    paths["hemi_map"] = pt
    return paths
