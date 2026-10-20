"""Seeded, deterministic synthetic surface generators returning ``pybiscuit.mesh.Mesh``.

All meshes are closed (except ``cut_sphere``), outward-oriented (counter-clockwise
seen from outside, positive enclosed volume), float64 vertices / int32 faces.
Anything random takes an explicit ``seed`` and uses ``numpy.random.default_rng``
(PCG64, stable across platforms/NumPy versions for the same bit generator).

Real spherical harmonics convention (``real_sph_harm``), with Y_l^m the complex
harmonic *including* the Condon-Shortley phase (as scipy returns it):

    m == 0:  Y_l0
    m  > 0:  sqrt(2) * (-1)**m * Re(Y_l^m)   (~ cos(m*phi) P_l^|m| without CS phase)
    m  < 0:  sqrt(2) * (-1)**m * Im(Y_l^|m|) (~ sin(|m|*phi) P_l^|m| without CS phase)

theta is the polar angle from +z, phi the azimuth from +x toward +y. These are
orthonormal on the unit sphere.

Cross-platform hashing (``geometry_hash``): vertices are quantised to integers
``rint(v * 1e9)`` (int64, little-endian) before hashing, which removes
last-bit libm differences (sin/cos/sqrt) and the -0.0 vs 0.0 ambiguity. Faces
are hashed as little-endian int32. Face/vertex order is part of the contract.
"""

from __future__ import annotations

import hashlib

import numpy as np

from pybiscuit.mesh import Mesh

try:  # scipy >= 1.15
    from scipy.special import sph_harm_y as _sph_harm_y

    def _ylm(l: int, m: int, theta, phi):  # noqa: E741
        return _sph_harm_y(l, m, theta, phi)

except ImportError:  # pragma: no cover - older scipy
    from scipy.special import sph_harm as _sph_harm

    def _ylm(l: int, m: int, theta, phi):  # noqa: E741
        return _sph_harm(m, l, phi, theta)


def real_sph_harm(l: int, m: int, theta, phi) -> np.ndarray:  # noqa: E741
    """Real orthonormal spherical harmonic (convention in the module docstring)."""
    if abs(m) > l:
        raise ValueError("|m| must be <= l")
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    if m == 0:
        return np.real(_ylm(l, 0, theta, phi))
    y = _ylm(l, abs(m), theta, phi)
    part = np.real(y) if m > 0 else np.imag(y)
    return np.sqrt(2.0) * (-1.0) ** m * part


def geometry_hash(mesh: Mesh) -> str:
    """SHA-256 of quantised vertices (1e-9) and faces; stable across platforms."""
    q = np.rint(mesh.vertices * 1e9).astype("<i8")
    f = np.ascontiguousarray(mesh.faces, dtype="<i4")
    h = hashlib.sha256()
    h.update(q.tobytes())
    h.update(f.tobytes())
    return h.hexdigest()


def uv_sphere(n_lat: int = 16, n_lon: int = 32, radius: float = 1.0) -> Mesh:
    """Latitude/longitude sphere: 2 poles + (n_lat-1) rings of n_lon vertices.

    Vertex 0 is the north pole (+z), vertex 1 the south pole, then rings north to south.
    """
    if n_lat < 2 or n_lon < 3:
        raise ValueError("need n_lat >= 2 and n_lon >= 3")
    theta = np.pi * np.arange(1, n_lat) / n_lat
    phi = 2.0 * np.pi * np.arange(n_lon) / n_lon
    st, ct = np.sin(theta)[:, None], np.cos(theta)[:, None]
    ring = np.stack(
        [
            (st * np.cos(phi)[None, :]).ravel(),
            (st * np.sin(phi)[None, :]).ravel(),
            np.broadcast_to(ct, (n_lat - 1, n_lon)).ravel(),
        ],
        axis=1,
    )
    verts = np.vstack([[[0.0, 0.0, 1.0]], [[0.0, 0.0, -1.0]], ring]) * radius

    def idx(i, j):  # ring i (0-based), longitude j
        return 2 + i * n_lon + (j % n_lon)

    faces = []
    for j in range(n_lon):
        faces.append((0, idx(0, j), idx(0, j + 1)))
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a, b = idx(i, j), idx(i, j + 1)
            c, d = idx(i + 1, j), idx(i + 1, j + 1)
            faces.append((a, c, b))
            faces.append((b, c, d))
    for j in range(n_lon):
        faces.append((1, idx(n_lat - 2, j + 1), idx(n_lat - 2, j)))
    return Mesh(verts, np.array(faces, dtype=np.int32), frame="RAS", units="mm")


def ellipsoid(
    a: float = 3.0, b: float = 2.0, c: float = 1.0, n_lat: int = 24, n_lon: int = 48
) -> Mesh:
    """Axis-aligned ellipsoid (semi-axes a, b, c) from the UV-sphere topology."""
    m = uv_sphere(n_lat, n_lon)
    m.vertices = m.vertices * np.array([a, b, c])
    return m


def icosphere(level: int = 0, radius: float = 1.0) -> Mesh:
    """Unit icosphere by simple midpoint subdivision, level 0..7.

    NOT the BISCUIT vertex/face ordering (that lives in pybiscuit.sphere.icosphere);
    this is a self-contained fixture. Counts: V = 10*4**level + 2, F = 20*4**level.
    """
    if not 0 <= level <= 7:
        raise ValueError("level must be in 0..7")
    t = (1.0 + np.sqrt(5.0)) / 2.0
    v = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    f = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )  # fmt: skip
    for _ in range(level):
        n = len(v)
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        e.sort(axis=1)
        uniq, inv = np.unique(e, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        mid = v[uniq[:, 0]] + v[uniq[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        v = np.vstack([v, mid])
        nf = len(f)
        m01, m12, m20 = (n + inv[k * nf : (k + 1) * nf] for k in range(3))
        a, b, c = f[:, 0], f[:, 1], f[:, 2]
        f = np.concatenate(
            [
                np.stack([a, m01, m20], 1),
                np.stack([b, m12, m01], 1),
                np.stack([c, m20, m12], 1),
                np.stack([m01, m12, m20], 1),
            ]
        )
    return Mesh(v * radius, f.astype(np.int32), frame="RAS", units="mm")


def folded_sphere(a: float = 0.1, l: int = 4, m: int = 2, level: int = 5) -> Mesh:  # noqa: E741
    """Analytic folded sphere r(theta, phi) = 1 + a * Y_lm on an icosphere(level).

    Y_lm is the real harmonic from ``real_sph_harm``. Vertex i of the result is
    ``u_i * r(u_i)`` for vertex i of ``icosphere(level)``; the ``point_data``
    entries ``"ylm"`` (Y_lm) and ``"radius"`` give the analytic values.
    """
    base = icosphere(level)
    u = base.vertices
    theta = np.arccos(np.clip(u[:, 2], -1.0, 1.0))
    phi = np.arctan2(u[:, 1], u[:, 0])
    y = real_sph_harm(l, m, theta, phi)
    r = 1.0 + a * y
    return Mesh(
        u * r[:, None],
        base.faces,
        point_data={"ylm": y, "radius": r},
        frame="RAS",
        units="mm",
        meta={"a": a, "l": l, "m": m, "level": level},
    )


def cut_sphere(level: int = 3, z_cut: float = 0.5) -> Mesh:
    """Open sphere: icosphere(level) minus faces whose centroid has z > z_cut.

    Unused vertices are dropped and the rest renumbered in original order, giving a
    single boundary loop (a spherical cap removed at the north pole).
    """
    m = icosphere(level)
    keep = m.vertices[m.faces].mean(axis=1)[:, 2] <= z_cut
    faces = m.faces[keep]
    used = np.unique(faces)
    remap = np.full(m.n_vertices, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return Mesh(m.vertices[used], remap[faces].astype(np.int32), frame="RAS", units="mm")


def perturb(mesh: Mesh, sigma: float, seed: int) -> Mesh:
    """Copy of ``mesh`` with i.i.d. Gaussian vertex noise (std ``sigma``), seeded."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(mesh.vertices.shape) * sigma
    return Mesh(mesh.vertices + noise, mesh.faces.copy(), frame=mesh.frame, units=mesh.units)
