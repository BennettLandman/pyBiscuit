# Derived from SphericalLabeling/meshcnn (Ilwoo Lyu, MIT), itself derived from
# UGSCNN MeshConv (https://github.com/maxjiang93/ugscnn, commit
# 64e27969c65b4807e41c1e275972dab8bb72c512).
#
# UGSCNN: Spherical CNNs on Unstructured Grids
# The MIT License (MIT)
# Copyright (c) 2018 Chiyu "Max" Jiang, University of California, Berkeley
# Full license text: LICENSES/UGSCNN-MeshConv.txt
# SphericalLabeling license (Copyright (c) 2019 Ilwoo Lyu): LICENSES/SphericalLabeling.txt
"""MeshConv (UGSCNN / SphericalLabeling) spherical operators.

The parcellation network (``ctf_plain/SurfPostproc/common/parc``) loads
``icosphere_{0..5}.pkl`` files holding, per icosphere level, the vertex and
face arrays plus FEM operators.  This module

* reads those third-party pickles *safely* (:func:`safe_load_pickle`);
* regenerates every stored field from first principles
  (:func:`build_operators`), as a pure NumPy/SciPy re-implementation of
  ``SphericalLabeling/meshcnn/mesh.py`` (MIT) with the libigl calls
  (``grad``, ``cotmatrix``, ``per_face_normals``, ``doublearea``,
  ``massmatrix``) replaced by NumPy code; and
* stores/loads operators as ``.npz`` without pickle
  (:func:`save_operators_npz`, :func:`load_operators_npz`).

Stored fields (keys of the original dict), level ``l`` with
``nf = 20*4**l`` faces, ``nv = 10*4**l + 2`` vertices:

``V`` (nv,3) f8, ``F`` (nf,3) int, ``nv_prev``/``nv_next`` int,
``G`` (3nf, nv) coo (gradient; x-, y-, z-component blocks stacked),
``L`` (nv, nv) coo (cotangent Laplacian), ``N`` (nf,3) face normals,
``NS``/``EW`` (nf,3) north-south / east-west unit-ish fields,
``F2V`` (nv, nf) csc (area-weighted face->vertex), ``M`` (nv,) Voronoi mass,
``Seq`` tuple of 11 permutations, ``Intp`` (nv, nv_prev) csc interpolation.

Security: the shipped pickles come from a third-party container image.  They
must never be passed to ``pickle.load``.  See ``docs/ports/meshconv-operators.md``.
"""

from __future__ import annotations

import io
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sp

__all__ = [
    "FIELDS",
    "SPARSE_FIELDS",
    "SafeUnpicklingError",
    "build_operators",
    "compare_operators",
    "load_operators_npz",
    "safe_load_pickle",
    "save_operators_npz",
]

FIELDS = ("V", "F", "nv_prev", "nv_next", "G", "L", "N", "NS", "EW", "F2V", "M", "Seq", "Intp")
SPARSE_FIELDS = ("G", "L", "F2V", "Intp")


# --------------------------------------------------------------------------
# Safe pickle loading
# --------------------------------------------------------------------------


class SafeUnpicklingError(pickle.UnpicklingError):
    """Raised when a pickle references a global outside the allowlist."""


def _np_reconstruct():
    mod = getattr(np, "_core", None) or np.core  # numpy>=2 renamed numpy.core
    return mod.multiarray._reconstruct


def _safe_codecs_encode(obj, encoding="utf-8", errors="strict"):
    """``_codecs.encode`` restricted to ``str -> bytes`` latin1 (py2 numpy pickles)."""
    if not isinstance(obj, str) or encoding != "latin1" or errors != "strict":
        raise SafeUnpicklingError("only _codecs.encode(str, 'latin1') is allowed")
    return obj.encode("latin1")


# Exact (module, name) pairs seen in `pickletools.dis` of icosphere_{0..5}.pkl
# (FC v2.2 image).  Everything else is rejected.
_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"): _np_reconstruct,
    ("numpy", "ndarray"): lambda: np.ndarray,
    ("numpy", "dtype"): lambda: np.dtype,
    ("_codecs", "encode"): lambda: _safe_codecs_encode,
    ("scipy.sparse.csc", "csc_matrix"): lambda: sp.csc_matrix,
    ("scipy.sparse.coo", "coo_matrix"): lambda: sp.coo_matrix,
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            getter = _ALLOWED[(module, name)]
        except KeyError:
            raise SafeUnpicklingError(f"global {module}.{name} is not allowed") from None
        return getter()

    def persistent_load(self, pid):
        raise SafeUnpicklingError("persistent ids are not allowed")


def safe_loads(data: bytes):
    """Unpickle ``data`` with the restricted allowlist."""
    return _RestrictedUnpickler(io.BytesIO(data)).load()


def safe_load_pickle(path):
    """Load a MeshConv ``icosphere_*.pkl`` through the restricted unpickler.

    Only the reconstructors observed in the shipped files are permitted
    (numpy ``_reconstruct``/``ndarray``/``dtype``, ``scipy.sparse`` csc/coo
    matrices, and a latin1-only ``_codecs.encode``).  Any other global raises
    :class:`SafeUnpicklingError` before it can be resolved or called.
    """
    with open(Path(path), "rb") as fh:
        return _RestrictedUnpickler(fh).load()


# --------------------------------------------------------------------------
# Icosphere generation (port of meshcnn/mesh.py + mesh_utils.py)
# --------------------------------------------------------------------------


def _float_to_int(data, digits=8):
    data = np.asanyarray(data)
    data_max = np.abs(data).max() * 10**digits
    dtype = [np.int32, np.int64][int(data_max > 2**31)]
    return np.round((data * 10**digits) - 1e-6).astype(dtype)


def _hashable_rows(data, digits=8):
    as_int = _float_to_int(data, digits)
    precision = int(np.floor(64 / as_int.shape[1]))
    if np.abs(as_int).max() < 2 ** (precision - 1):
        hashable = np.zeros(len(as_int), dtype=np.int64)
        for offset, column in enumerate(as_int.astype(np.int64).T):
            np.bitwise_xor(hashable, column << (offset * precision), out=hashable)
        return hashable
    dtype = np.dtype((np.void, as_int.dtype.itemsize * as_int.shape[1]))
    return np.ascontiguousarray(as_int).view(dtype).reshape(-1)


def _unique_rows(data):
    _, unique, inverse = np.unique(_hashable_rows(data), return_index=True, return_inverse=True)
    return unique, inverse.reshape(-1)


def _icosahedron():
    t = (1.0 + 5.0**0.5) / 2.0
    vertices = [-1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t,
                0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1]  # fmt: skip
    faces = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
             1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
             3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
             4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]  # fmt: skip
    vertices = np.reshape(vertices, (-1, 3)) / 1.9021130325903071
    faces = np.reshape(faces, (-1, 3))
    return vertices, faces


def _subdivide(vertices, faces):
    face_index = np.arange(len(faces))
    triangles = vertices[faces]
    pairs = [[0, 1], [1, 2], [2, 0]]
    src_idx = np.vstack([faces[:, g] for g in pairs])
    mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in pairs])
    mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
    unique, inverse = _unique_rows(mid)
    mid = mid[unique]
    src_idx = src_idx[unique]
    mid_idx = inverse[mid_idx] + len(vertices)
    f = np.column_stack([
        faces[:, 0], mid_idx[:, 0], mid_idx[:, 2],
        mid_idx[:, 0], faces[:, 1], mid_idx[:, 1],
        mid_idx[:, 2], mid_idx[:, 1], faces[:, 2],
        mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2],
    ]).reshape((-1, 3))  # fmt: skip
    new_faces = np.vstack((faces, f[len(face_index) :]))
    new_faces[face_index] = f[: len(face_index)]
    new_vertices = np.vstack((vertices, mid))
    nv = vertices.shape[0]
    identity_map = np.stack((np.arange(nv), np.arange(nv)), axis=1)
    src_id = np.concatenate((identity_map, src_idx), axis=0)
    return new_vertices, new_faces, src_id


def _normalize(vertices, radius=1.0):
    scalar = (vertices**2).sum(axis=1) ** 0.5
    unit = vertices / scalar.reshape((-1, 1))
    offset = radius - scalar
    return vertices + unit * offset.reshape((-1, 1))


def _rot_matrix(rot_axis, cos_t, sin_t):
    k = rot_axis / np.linalg.norm(rot_axis)
    eye = np.eye(3)
    cols = []
    for i in range(3):
        v = eye[i]
        cols.append(v * cos_t + np.cross(k, v) * sin_t + k * (k.dot(v)) * (1 - cos_t))
    return np.stack(cols, axis=-1)


def _find_neighbor(faces, ind):
    ff = np.unique(np.concatenate([f for f in faces if ind in f]))
    return [f for f in ff if f != ind]


def _ico_rot_matrix(v0, f0, ind):
    v0_ = v0.copy()
    top = v0_[ind]
    z0 = np.array([0, 0, 1])
    k = np.cross(top, z0)
    ct = np.dot(top, z0)
    st = -np.linalg.norm(k)
    r1 = _rot_matrix(k, ct, st)
    v0_ = v0_.dot(r1)
    ni = _find_neighbor(f0, ind)[0]
    vec = v0_[ni].copy()
    vec[2] = 0
    vec = vec / np.linalg.norm(vec)
    y_ = np.eye(3)[1]
    k = np.eye(3)[2]
    crs = np.cross(vec, y_)
    ct = np.dot(vec, y_)
    st = -np.sign(crs[-1]) * np.linalg.norm(crs)
    r2 = _rot_matrix(k, ct, st)
    return r1.dot(r2)


def _rotseq(v, v0, f0, acc=9):
    seq = []
    for i in range(11):
        vr = v.dot(_ico_rot_matrix(v0, f0, i))
        s1 = np.lexsort(np.round(v.T, acc))
        s2 = np.lexsort(np.round(vr.T, acc))
        seq.append(s1[np.argsort(s2)])
    return tuple(seq)


def _icosphere_mesh(level):
    """Return ``(V, F, intp, V0, F0)`` exactly as ``meshcnn.mesh.icosphere``."""
    v, f = _icosahedron()
    v0, f0 = v.copy(), f.copy()
    intp = None
    for _ in range(level):
        v, f, intp = _subdivide(v, f)
        v = _normalize(v)
    return v, f, intp, v0, f0


# --------------------------------------------------------------------------
# libigl replacements
# --------------------------------------------------------------------------


def _face_geometry(v, f):
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cr = np.cross(p1 - p0, p2 - p0)
    dbl = np.linalg.norm(cr, axis=1)
    return p0, p1, p2, cr, dbl


def _grad(v, f):
    """Per-face gradient operator (``igl.grad``): (3nf, nv), blocks x|y|z."""
    nf, nv = f.shape[0], v.shape[0]
    p0, p1, p2, cr, dbl = _face_geometry(v, f)
    n = cr / dbl[:, None]
    # gradient of the hat function of vertex i: n x (opposite edge) / (2 A)
    opp = (p2 - p1, p0 - p2, p1 - p0)
    rows, cols, vals = [], [], []
    ar = np.arange(nf)
    for i in range(3):
        g = np.cross(n, opp[i]) / dbl[:, None]  # (nf, 3)
        for d in range(3):
            rows.append(ar + d * nf)
            cols.append(f[:, i])
            vals.append(g[:, d])
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * nf, nv),
    )


def _cotmatrix(v, f):
    """Cotangent Laplacian (``igl.cotmatrix``): off-diagonals >= 0, diagonal = -row sum."""
    nv = v.shape[0]
    p = (v[f[:, 0]], v[f[:, 1]], v[f[:, 2]])
    dbl = _face_geometry(v, f)[4]
    rows, cols, vals = [], [], []
    for k in range(3):  # angle at vertex k, opposite edge (i, j)
        i, j = (k + 1) % 3, (k + 2) % 3
        a, b = p[i] - p[k], p[j] - p[k]
        cot = (a * b).sum(axis=1) / dbl  # cot = cos/sin = a.b / |a x b|
        w = 0.5 * cot
        rows += [f[:, i], f[:, j], f[:, i], f[:, j]]
        cols += [f[:, j], f[:, i], f[:, i], f[:, j]]
        vals += [w, w, -w, -w]
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nv, nv)
    )


def _voronoi_mass(v, f):
    """Diagonal (Meyer et al. mixed) Voronoi mass, per vertex (``igl.massmatrix``)."""
    nv = v.shape[0]
    p = (v[f[:, 0]], v[f[:, 1]], v[f[:, 2]])
    dbl = _face_geometry(v, f)[4]
    area = 0.5 * dbl
    l2 = np.stack([((p[(k + 1) % 3] - p[(k + 2) % 3]) ** 2).sum(axis=1) for k in range(3)], 1)
    dots = np.stack(
        [((p[(k + 1) % 3] - p[k]) * (p[(k + 2) % 3] - p[k])).sum(axis=1) for k in range(3)], 1
    )
    cot = dots / dbl[:, None]  # cot of angle at corner k
    obtuse = (dots < 0).any(axis=1)
    m = np.zeros(nv)
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        # classic Voronoi region of corner i / j from angle at k's opposite edges
        # corner k gets (l2[i_edge]*cot[j] + l2[j_edge]*cot[i]) / 8 with edges adjacent to k
        e_adj1 = l2[:, j]  # edge (k, i) has length^2 = l2[j]
        e_adj2 = l2[:, i]  # edge (k, j) has length^2 = l2[i]
        vor = (e_adj1 * cot[:, j] + e_adj2 * cot[:, i]) / 8.0
        mix = np.where(dots[:, k] < 0, area / 2.0, area / 4.0)
        w = np.where(obtuse, mix, vor)
        np.add.at(m, f[:, k], w)
    return m


# --------------------------------------------------------------------------
# build_operators
# --------------------------------------------------------------------------


def build_operators(level: int) -> dict:
    """Regenerate the MeshConv operator dict for icosphere ``level`` (0..7).

    Returns the same keys as the shipped pickles (:data:`FIELDS`).  Sparse
    ``G``, ``L`` are ``coo_matrix``; ``F2V``, ``Intp`` are ``csc_matrix``
    (the container types of the originals).
    """
    level = int(level)
    if level < 0:
        raise ValueError("level must be >= 0")
    v, f, intp, v0, f0 = _icosphere_mesh(level)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    lat = np.arctan2(z, np.sqrt(x**2 + y**2))
    nf = 20 * 4**level
    ne = 30 * 4**level
    nv = ne - nf + 2
    nv_prev = int(ne / 4 - nf / 4 + 2)
    nv_next = int(ne * 4 - nf * 4 + 2)
    assert nv == v.shape[0] and nf == f.shape[0]

    g = _grad(v, f)
    lap = _cotmatrix(v, f)
    _, _, _, cr, dbl = _face_geometry(v, f)
    nrm = cr / dbl[:, None]
    mass = _voronoi_mass(v, f)

    ns = np.reshape(g.tocsr().dot(lat), [nf, 3], order="F")
    ew = np.cross(ns, nrm)

    i = f.ravel()
    j = np.arange(nf).repeat(3)
    adj = sp.csc_matrix((np.ones(nf * 3), (i, j)), shape=(nv, nf))
    tot_area = adj.dot(dbl)
    norm_area = dbl.repeat(3) / np.squeeze(tot_area[i])
    f2v = sp.csc_matrix((norm_area, (i, j)), shape=(nv, nf))

    if level > 0:
        ip = intp[nv_prev:]
        ii = np.concatenate((np.arange(nv), np.arange(nv_prev, nv)))
        jj = np.concatenate((np.arange(nv_prev), ip[:, 0], ip[:, 1]))
        ratio = np.concatenate((np.ones(nv_prev), 0.5 * np.ones(2 * ip.shape[0])))
        interp = sp.csc_matrix((ratio, (ii, jj)), shape=(nv, nv_prev))
    else:
        interp = sp.csc_matrix(np.eye(nv))

    return {
        "V": v,
        "F": f,
        "nv_prev": nv_prev,
        "nv_next": nv_next,
        "G": g,
        "L": lap,
        "N": nrm,
        "NS": ns,
        "EW": ew,
        "F2V": f2v,
        "M": mass,
        "Seq": _rotseq(v, v0, f0),
        "Intp": interp,
    }


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _pattern(m):
    c = m.tocoo()
    key = np.unique(c.row.astype(np.int64) * m.shape[1] + c.col)
    return key


def compare_operators(a: dict, b: dict) -> dict:
    """Field-by-field comparison.  Returns ``{field: (max_abs_diff, note)}``.

    Dense fields: max abs difference (``inf`` on shape mismatch).  Sparse
    fields: max abs difference of the (canonicalised) values on the union
    pattern, with ``note`` listing pattern differences (explicit zeros are
    ignored when judging the nonzero pattern but reported in ``note``).
    ``Seq``: max abs diff of the integer permutations.
    """
    out = {}
    for k in FIELDS:
        x, y = a[k], b[k]
        if k in ("nv_prev", "nv_next"):
            out[k] = (float(abs(int(x) - int(y))), "")
        elif k == "Seq":
            if len(x) != len(y):
                out[k] = (np.inf, "length")
            else:
                d = max(
                    float(np.abs(np.asarray(p, float) - np.asarray(q, float)).max())
                    if np.shape(p) == np.shape(q)
                    else np.inf
                    for p, q in zip(x, y, strict=True)
                )
                out[k] = (d, "")
        elif k in SPARSE_FIELDS:
            if x.shape != y.shape:
                out[k] = (np.inf, f"shape {x.shape} vs {y.shape}")
                continue
            px, py = _pattern(x), _pattern(y)
            note = []
            if px.shape != py.shape or not np.array_equal(px, py):
                note.append(f"stored-pattern differs ({px.size} vs {py.size} unique entries)")
            xc = sp.csc_matrix(x)
            yc = sp.csc_matrix(y)
            xc.sum_duplicates()
            yc.sum_duplicates()
            d = abs(xc - yc)
            out[k] = (float(d.max()) if d.nnz else 0.0, "; ".join(note))
        else:
            x, y = np.asarray(x), np.asarray(y)
            if x.shape != y.shape:
                out[k] = (np.inf, f"shape {x.shape} vs {y.shape}")
            else:
                out[k] = (float(np.abs(x.astype(float) - y.astype(float)).max()), "")
    return out


# --------------------------------------------------------------------------
# npz (pickle-free) storage
# --------------------------------------------------------------------------


def save_operators_npz(path, ops: dict) -> None:
    """Write operators to a compressed ``.npz`` (no pickled objects)."""
    arrays: dict[str, np.ndarray] = {}
    for k in FIELDS:
        val = ops[k]
        if k in SPARSE_FIELDS:
            c = sp.coo_matrix(val)
            arrays[f"{k}__row"] = c.row.astype(np.int64)
            arrays[f"{k}__col"] = c.col.astype(np.int64)
            arrays[f"{k}__data"] = np.asarray(c.data, dtype=np.float64)
            arrays[f"{k}__shape"] = np.asarray(c.shape, dtype=np.int64)
            arrays[f"{k}__fmt"] = np.asarray(type(val).__name__)
        elif k == "Seq":
            arrays["Seq"] = np.stack([np.asarray(s, dtype=np.int64) for s in val])
        else:
            arrays[k] = np.asarray(val)
    np.savez_compressed(Path(path), **arrays)


def load_operators_npz(path) -> dict:
    """Load operators written by :func:`save_operators_npz` (``allow_pickle=False``)."""
    out: dict = {}
    with np.load(Path(path), allow_pickle=False) as z:
        for k in FIELDS:
            if k in SPARSE_FIELDS:
                m = sp.coo_matrix(
                    (z[f"{k}__data"], (z[f"{k}__row"], z[f"{k}__col"])),
                    shape=tuple(int(s) for s in z[f"{k}__shape"]),
                )
                out[k] = m.tocsc() if str(z[f"{k}__fmt"]).startswith("csc") else m
            elif k == "Seq":
                out[k] = tuple(np.asarray(s) for s in z["Seq"])
            elif k in ("nv_prev", "nv_next"):
                out[k] = int(z[k])
            else:
                out[k] = z[k]
    return out
