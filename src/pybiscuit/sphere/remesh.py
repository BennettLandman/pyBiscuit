"""Spherical remeshing / data resampling: port of ilwoolyu/SphericalRemesh.

Source: ``reference/repos/SphericalRemesh`` @ b677c39 (``src/SphericalRemeshing.cpp``,
``src/SphericalRemeshing.h``, ``wrapper/main.cpp``, ``wrapper/PARSE_ARGS.h``) with
the MeshLib (@ dcef807) pieces it uses (``Vector``/``Coordinate`` float32 helpers in
``Util/Geom.cpp``, ``VtkIO::save``).  See ``docs/ports/spherical-remesh.md``.

Public API
==========

:class:`SphericalRemeshing`
    Mirrors the C++ class one-to-one (constructor arguments, ``degree``,
    ``deform``, ``save_deformed_surface``, ``save_deformed_sphere``,
    ``save_deformed_property``, ``save_bary``) plus array accessors
    (``remeshed_vertices``, ``deformed_data``, ``colors``, ``bary()``,
    ``deformed_sphere_vertices``).
:func:`run`
    The body of ``wrapper/main.cpp`` (the ``--deg0..--deg`` output loop).
:func:`resample`
    Convenience: resample per-vertex arrays from one sphere onto another.

Precision: all geometry is float32 exactly like MeshLib (``float`` vertices);
the SH basis is evaluated in float64 and rounded to float32 (MeshLib's float
overload), the deformation sums are sequential float32.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numba import njit

from pybiscuit.io import read_surface
from pybiscuit.mesh import Mesh
from pybiscuit.sphere.aabb import SphereAABB
from pybiscuit.sphere.spharm import basis, read_sh_coefficients, synthesize

__all__ = [
    "FTYPES",
    "SphericalRemeshing",
    "check_ftype",
    "resample",
    "run",
]

F = np.float32
FTYPES = ("int", "int16", "int32", "float", "double")  # PARSE_ARGS.h --ftype members


def check_ftype(name: str) -> str:
    """CLI11 ``IsMember({...}, ignore_case, ignore_underscore)`` validation of ``--ftype``.

    Returns ``name`` *unchanged*: CLI11 1.9.1 validates case/underscore-insensitively but
    passes the user's spelling on, and the writer then compares with ``strcmp`` against the
    lower-case names.  ``Float`` / ``INT_16`` / ``DOUBLE`` therefore pass validation but
    produce empty ``.dat`` files (verified against the oracle); we reproduce that.
    """
    key = name.lower().replace("_", "")
    if key not in FTYPES:
        raise ValueError(f"{name!r} not in {{{','.join(FTYPES)}}}")
    return name


# --------------------------------------------------------------------------- #
# float32 scalar helpers (MathVector / Coordinate from MeshLib Geom.cpp)
# --------------------------------------------------------------------------- #
@njit(cache=True, inline="always")
def _sin(x):
    return F(math.sin(np.float64(x)))


@njit(cache=True, inline="always")
def _cos(x):
    return F(math.cos(np.float64(x)))


@njit(cache=True, inline="always")
def _unit3(x, y, z):
    """``MathVector::unit`` (no-op when the norm is not positive)."""
    s = F(math.sqrt(np.float64(x * x + y * y + z * z)))
    if s > F(0):
        return x / s, y / s, z / s
    return x, y, z


@njit(cache=True, inline="always")
def _cross3(ax, ay, az, bx, by, bz):
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


@njit(cache=True, inline="always")
def _sph2cart(phi, theta):
    """``Coordinate::sph2cart(float, float, float*)``: phi azimuth, theta elevation."""
    z = _sin(theta)
    ce = _cos(theta)
    return ce * _cos(phi), ce * _sin(phi), z


@njit(cache=True, inline="always")
def _rotation(ax, ay, az, theta):
    """``Coordinate::rotation(axis, theta, mat)``: Rodrigues, row-major 9 floats."""
    m = np.zeros(9, np.float32)
    m[0] = F(1)
    m[4] = F(1)
    m[8] = F(1)
    n = F(math.sqrt(np.float64(ax * ax + ay * ay + az * az)))
    if n == F(0):
        return m
    ax, ay, az = _unit3(ax, ay, az)
    A = np.empty(9, np.float32)
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
    return m


@njit(cache=True, inline="always")
def _rot_point(m, x, y, z):
    return (
        m[0] * x + m[1] * y + m[2] * z,
        m[3] * x + m[4] * y + m[5] * z,
        m[6] * x + m[7] * y + m[8] * z,
    )


@njit(cache=True)
def _recons_all(v0, delta, axis0, tan1, tan2, out):
    """Second half of ``SphericalRemeshing::reconsCoord`` for every vertex.

    ``v0`` (N,3) float32 start positions, ``delta`` (N,3) float32 SH sums.
    ``out`` receives the final *unnormalised* rotated points (``v1``).
    """
    for i in range(v0.shape[0]):
        d0, d1, d2 = delta[i, 0], delta[i, 1], delta[i, 2]
        # axis = unit(axis0 + tan1 * d1 + tan2 * d2)
        tx = axis0[0] + tan1[0] * d1
        ty = axis0[1] + tan1[1] * d1
        tz = axis0[2] + tan1[2] * d1
        tx = tx + tan2[0] * d2
        ty = ty + tan2[1] * d2
        tz = tz + tan2[2] * d2
        qx, qy, qz = _unit3(tx, ty, tz)
        angle = F(math.sqrt(np.float64(d1 * d1 + d2 * d2)))
        ax, ay, az = _cross3(axis0[0], axis0[1], axis0[2], qx, qy, qz)
        ax, ay, az = _unit3(ax, ay, az)
        rot = _rotation(ax, ay, az, angle)
        rx, ry, rz = _rot_point(rot, v0[i, 0], v0[i, 1], v0[i, 2])
        rot = _rotation(qx, qy, qz, d0)
        px, py, pz = _rot_point(rot, rx, ry, rz)
        out[i, 0] = px
        out[i, 1] = py
        out[i, 2] = pz


def _unit_rows(v: np.ndarray) -> np.ndarray:
    """Row-wise ``MathVector::unit`` in float32 (zero rows unchanged)."""
    v = np.ascontiguousarray(v, dtype=np.float32)
    s = np.sqrt(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2])
    ok = s > 0
    out = v.copy()
    out[ok] = v[ok] / s[ok, None]
    return out


# --------------------------------------------------------------------------- #
# file helpers
# --------------------------------------------------------------------------- #
def _load_mesh(m) -> tuple[np.ndarray, np.ndarray]:
    """float32 vertices (MeshLib ``float``) and int32 faces from a path or ``Mesh``."""
    if not isinstance(m, Mesh):
        m = read_surface(m)
    return m.vertices.astype(np.float32), np.ascontiguousarray(m.faces, dtype=np.int32)


def _fmt_rows(fmt: str, rows) -> str:
    return "\n".join(map(fmt.__mod__, map(tuple, rows))) + "\n"


def _write_vtk_meshlib(path, verts: np.ndarray, faces: np.ndarray) -> None:
    """``VtkIO::save`` (ASCII, no normals): ``%.9f`` fixed vertices, ``\\n`` newlines."""
    n, m = len(verts), len(faces)
    head = f"# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS {n} float\n"
    body = _fmt_rows("%.9f %.9f %.9f", verts.tolist()) if n else ""
    poly = f"POLYGONS {m} {m * 4}\n"
    fcs = _fmt_rows("3 %d %d %d", faces.tolist()) if m else ""
    Path(path).write_bytes((head + body + poly + fcs + "\n").encode())


def _read_floats(path, n: int) -> np.ndarray:
    toks = Path(path).read_text().split()
    if len(toks) < n:
        raise ValueError(f"{path}: expected {n} values, found {len(toks)} (Fatal error: I/O)")
    return np.array(toks[:n], dtype=np.float64).astype(np.float32)


def _property_name_surface(p: str) -> str:
    name = p[: max(len(p) - 4, 0)]
    found = name.rfind(".") + 1
    return name[found:]


def _property_name_file(p: str) -> str:
    name = p[: max(len(p) - 4, 0)]
    found = name.rfind(".") + 1
    if found == 0:
        found = name.rfind("_") + 1
    return name[found:]


# --------------------------------------------------------------------------- #
# the class
# --------------------------------------------------------------------------- #
class SphericalRemeshing:
    """Port of the C++ ``SphericalRemeshing`` class.

    Parameters mirror the C++ constructor (``wrapper/main.cpp`` passes
    ``interpolation = not --nneighbor`` and ``verbose = not --quiet``):

    subject
        ``-i`` subject surface (path/Mesh) or ``None``.
    sphere
        ``-s`` spherical model used for the original parameterisation (path/Mesh).
    dfield
        ``-c`` HSD coefficient file or ``None``.
    keep_color
        ``--keepColor``.
    sphere_t
        ``-r`` reference sphere (defaults to ``sphere``).
    colormap
        ``--color`` file (one ``r g b`` integer triple per reference vertex).
    property
        ``-p`` list; each entry a text-file path (one value per ``sphere`` vertex)
        or a ``(name, array)`` pair.  The output name of a path entry is derived
        from the file name exactly as upstream does.
    interpolation
        ``True``: barycentric; ``False``: nearest-neighbour (data) / median
        (surface coordinates).
    deg
        ``--deg`` cap on the spherical-harmonic degree (-1: none).
    pbtype
        ``--ftype`` string as given (exact lower-case names write data) or ``None`` (text).
    """

    def __init__(
        self,
        subject=None,
        sphere=None,
        dfield=None,
        keep_color: bool = False,
        sphere_t=None,
        colormap=None,
        property: Sequence = (),  # noqa: A002 - mirrors upstream name
        interpolation: bool = True,
        deg: int = -1,
        verbose: bool = True,
        pbtype: str | None = None,
        backward: bool = False,
    ) -> None:
        if sphere is None:
            raise ValueError("sphere is required")
        self._verbose = verbose
        self._keep_color = keep_color
        self._pbtype = pbtype
        self._backward = backward
        self._interpolation = interpolation
        self._has_subject = subject is not None

        def say(msg: str) -> None:
            if verbose:
                print(msg, end="")

        self._subj_xyz = None
        if subject is not None:
            say("loading subject surface model..\n")
            sv, _ = _load_mesh(subject)
            self._subj_xyz = sv  # m_x, m_y, m_z

        say("Loading unit sphere information..\n")
        sv, sf = _load_mesh(sphere)
        self._sphere_subj_faces = sf
        self._v0_subj = _unit_rows(sv)  # normalised original
        self._sphere_subj_v = self._v0_subj.copy()  # m_sphere_subj (deformed in place)
        tv, tf = _load_mesh(sphere_t if sphere_t is not None else sphere)
        self._sphere_faces = tf
        self._sphere_v = _unit_rows(tv)  # m_sphere (unit)
        self._remesh_v = tv.copy()  # m_remesh (raw ref, overwritten by deformSurface)
        self._remesh_faces = tf

        # spherical harmonics information
        if dfield is not None:
            say("Loading spherical harmonics information..\n")
            d = read_sh_coefficients(dfield, max_degree=deg, dtype=np.float32)
            self._degree = d.degree
            self._pole = d.pole.astype(np.float32)
            self._coeff = d.coeff
        else:
            self._pole = np.zeros(2, np.float32)
            self._degree = 0
            self._coeff = np.zeros((3, 1), np.float32)
        if dfield is None and deg >= 0:
            self._degree = min(self._degree, deg)

        # tangent basis (SphericalRemeshing.cpp:149-156)
        p0, p1 = self._pole
        fmean = np.array(_sph2cart(p0, p1), dtype=np.float32)
        t1 = np.array(_sph2cart(F(np.float64(p0) + 0.1), F(np.float64(p1) + 0.1)), dtype=np.float32)
        tan1 = np.array(_unit3(*(t1 - fmean)), dtype=np.float32)
        c = _cross3(*fmean, *tan1)
        tan2 = np.array(_unit3(*c), dtype=np.float32)
        self._axis0, self._tan1, self._tan2 = fmean, tan1, tan2

        say("Initializing deformation..\n")
        base = self._sphere_v if backward else self._sphere_subj_v
        self._v0 = base.copy()
        self._Y = basis(np.ascontiguousarray(base), self._degree)

        # properties
        self._property_paths: list[str] = []
        self._ref_maps: list[np.ndarray] = []
        self._prop_names: list[tuple[str, str]] = []  # (surface-name, file-name)
        if len(property):
            say("Load properties..\n")
            say(f"\t{len(property)} found!\n")
            nprop = len(self._v0_subj)
            for item in property:
                if isinstance(item, tuple):
                    name, arr = item
                    arr = np.asarray(arr, dtype=np.float32)
                    if arr.shape != (nprop,):
                        raise ValueError(f"property {name!r}: expected {nprop} values")
                    self._property_paths.append(str(name) + ".xxx")  # 4 chars stripped
                    self._ref_maps.append(arr.copy())
                else:
                    say(f"\t{item}\n")
                    self._property_paths.append(str(item))
                    self._ref_maps.append(_read_floats(item, nprop))

        say("Tree initialization..\n")
        self._tree = SphereAABB(self._sphere_subj_v, self._sphere_subj_faces)

        # colours (constructor reads one triple per *reference* vertex)
        self._color: np.ndarray | None = None
        self._color_base: np.ndarray | None = None
        if colormap is not None:
            say("Loading color information..\n")
            n = len(self._remesh_v)
            toks = Path(colormap).read_text().split()
            if len(toks) < 3 * n:
                raise ValueError(f"{colormap}: expected {n} colour triples (Fatal error: I/O)")
            col = np.array(toks[: 3 * n], dtype=np.float64).astype(np.int32).reshape(n, 3)
            if keep_color:
                self._color_base = col
            else:
                self._color = col

        self._deformed_data: list[np.ndarray] = []
        self._bary_cache: tuple[np.ndarray, np.ndarray] | None = None

    # ---------------------------------------------------------------- accessors
    def degree(self) -> int:
        """Maximum SH degree actually in use (``m_degree``)."""
        return self._degree

    @property
    def remeshed_vertices(self) -> np.ndarray:
        """float32 vertices of the remeshed surface (``m_remesh``)."""
        return self._remesh_v

    @property
    def remeshed_faces(self) -> np.ndarray:
        return self._remesh_faces

    @property
    def deformed_sphere_vertices(self) -> np.ndarray:
        """float32 vertices of the (deformed) subject sphere (``m_sphere_subj``)."""
        return self._sphere_subj_v

    @property
    def deformed_data(self) -> list[np.ndarray]:
        """Per-property float32 arrays on the reference sphere (``m_deData``)."""
        return self._deformed_data

    @property
    def colors(self) -> np.ndarray | None:
        """Per-reference-vertex int32 RGB (``m_color``) or ``None``."""
        return self._color

    @property
    def property_names(self) -> list[str]:
        """Names used in ``saveDeformedProperty`` (``<prefix>.<name>.txt``)."""
        return [_property_name_file(p) for p in self._property_paths]

    def bary(self) -> tuple[np.ndarray, np.ndarray]:
        """``(face, coeff)``: closest face of the deformed sphere per reference vertex."""
        if self._bary_cache is None:
            face, coeff = self._tree.closest_face(np.ascontiguousarray(self._sphere_v))
            if (face < 0).any():
                raise RuntimeError(f"{int((face < 0).sum())} reference vertices not located")
            self._bary_cache = (face, coeff)
        return self._bary_cache

    # ---------------------------------------------------------------- deformation
    def deform(self, degree: int) -> None:
        """``SphericalRemeshing::deform``: apply the SH deformation of degree ``degree``."""
        degree = (self._degree if degree > self._degree else degree) if degree >= 0 else 0
        delta = synthesize(self._Y, self._coeff, degree=degree)
        v1 = np.empty_like(self._v0)
        _recons_all(self._v0, delta, self._axis0, self._tan1, self._tan2, v1)
        v = _unit_rows(v1)
        if self._backward:
            self._sphere_v = v
        else:
            self._sphere_subj_v = v
        self._tree.update(self._sphere_subj_v)
        self._bary_cache = None
        if self._has_subject:
            if self._verbose:
                print("Remeshing..")
            self._deform_surface()
        if self._ref_maps:
            if self._verbose:
                print("Property transferring..")
            self._deform_data()

    def _interp(self, ref: np.ndarray, face, coeff) -> np.ndarray:
        f = self._sphere_subj_faces[face]
        return ref[f[:, 0]] * coeff[:, 0] + ref[f[:, 1]] * coeff[:, 1] + ref[f[:, 2]] * coeff[:, 2]

    def _deform_surface(self) -> None:
        self._color = None
        face, coeff = self.bary()
        f = self._sphere_subj_faces[face]
        xyz = self._subj_xyz
        if self._interpolation:
            out = np.empty((len(face), 3), np.float32)
            for k in range(3):
                out[:, k] = self._interp(np.ascontiguousarray(xyz[:, k]), face, coeff)
            if self._keep_color:
                cb = self._color_base
                if cb is None or len(cb) < len(self._v0_subj):
                    raise ValueError("--keepColor needs a --color file covering the -s sphere")
                col = np.empty((len(face), 3), np.int32)
                for k in range(3):
                    ck = cb[:, k].astype(np.float32)
                    col[:, k] = self._interp(ck, face, coeff).astype(np.int32)  # (int) truncation
                self._color = col
        else:
            tri = xyz[f]  # (n, 3 verts, 3 coords)
            out = np.sort(tri, axis=1)[:, 1, :]  # Statistics::median of 3
        self._remesh_v = np.ascontiguousarray(out, dtype=np.float32)

    def _deform_data(self) -> None:
        face, coeff = self.bary()
        f = self._sphere_subj_faces[face]
        res = []
        for ref in self._ref_maps:
            if self._interpolation:
                res.append(self._interp(ref, face, coeff).astype(np.float32))
            else:
                a, b, c = ref[f[:, 0]], ref[f[:, 1]], ref[f[:, 2]]
                c0, c1, c2 = coeff[:, 0], coeff[:, 1], coeff[:, 2]
                pick_a = (c0 > c1) & (c0 > c2)
                pick_b = ~pick_a & (c1 > c2) & (c1 > c0)
                res.append(np.where(pick_a, a, np.where(pick_b, b, c)).astype(np.float32))
        self._deformed_data = res

    # ---------------------------------------------------------------- output
    def save_deformed_surface(self, filename) -> None:
        """``saveDeformedSurface``: remeshed VTK (+ ``COLOR_SCALARS``, ``FIELD ScalarData``)."""
        _write_vtk_meshlib(filename, self._remesh_v, self._remesh_faces)
        nv = len(self._sphere_v)
        parts = []
        have_color = self._color is not None and len(self._color) == nv
        if have_color:
            parts.append(f"POINT_DATA {nv}\nCOLOR_SCALARS colors 3\n")
            parts.append(_fmt_rows("%f %f %f", (self._color / 255.0).tolist()))
        if self._ref_maps:
            if not have_color:
                parts.append(f"POINT_DATA {nv}\n")
            parts.append(f"FIELD ScalarData {len(self._ref_maps)}\n")
            for i, p in enumerate(self._property_paths):
                parts.append(f"{_property_name_surface(p)} 1 {nv} float\n")
                parts.append("".join(map("%f\n".__mod__, self._deformed_data[i].tolist())))
                parts.append("\n")
        if parts:
            with open(filename, "ab") as fh:
                fh.write("".join(parts).encode())

    def save_deformed_sphere(self, filename) -> None:
        """``saveDeformedSphere``: the (deformed, normalised) ``-s`` sphere as VTK."""
        _write_vtk_meshlib(filename, self._sphere_subj_v, self._sphere_subj_faces)

    def save_deformed_property(self, filename) -> None:
        """``saveDeformedProperty``: ``<filename>.<name>.txt`` (or ``.dat`` with ``--ftype``)."""
        for i, p in enumerate(self._property_paths):
            ext = "txt" if self._pbtype is None else "dat"
            full = f"{filename}.{_property_name_file(p)}.{ext}"
            d = self._deformed_data[i]
            t = self._pbtype
            if t is None:
                Path(full).write_bytes("".join(map("%f\n".__mod__, d.tolist())).encode())
            elif t == "float":
                Path(full).write_bytes(d.astype("<f4").tobytes())
            elif t == "int16":
                Path(full).write_bytes(d.astype("<i2").tobytes())
            elif t in ("int", "int32"):
                Path(full).write_bytes(d.astype("<i4").tobytes())
            elif t == "double":
                Path(full).write_bytes(d.astype("<f8").tobytes())
            else:  # unknown type: upstream opens the file and writes nothing
                Path(full).write_bytes(b"")

    def save_bary(self, filename) -> None:
        """``saveBary``: ``<face> <c0> <c1> <c2>`` per reference vertex (``%d %f %f %f``)."""
        face, coeff = self.bary()
        lines = (
            f"{i:d} {a:f} {b:f} {c:f}\n"
            for i, a, b, c in zip(face.tolist(), *coeff.T.tolist(), strict=True)
        )
        Path(filename).write_bytes("".join(lines).encode())


# --------------------------------------------------------------------------- #
# main() loop and convenience wrapper
# --------------------------------------------------------------------------- #
def run(
    sphere,
    input=None,  # noqa: A002 - mirrors the CLI flag
    coeff=None,
    ref=None,
    property: Sequence = (),  # noqa: A002
    color=None,
    output=None,
    output_property=None,
    deform=None,
    bary=None,
    ftype: str | None = None,
    nneighbor: bool = False,
    deg: int = -1,
    deg0: int = -1,
    keep_color: bool = False,
    quiet: bool = False,
) -> None:
    """Equivalent of ``wrapper/main.cpp`` (all flags; outputs are written to disk)."""
    pb = check_ftype(ftype) if ftype else None
    sr = SphericalRemeshing(
        input if input else None,
        sphere,
        coeff if coeff else None,
        keep_color,
        ref if ref else None,
        color if color else None,
        list(property),
        not nneighbor,
        deg,
        not quiet,
        pb,
    )
    deg = sr.degree()
    if deg0 == -1 or deg0 > deg:
        deg0 = deg

    def say(msg: str) -> None:
        if not quiet:
            print(msg)

    for i in range(deg0, deg + 1):
        sr.deform(i)
        suffix = str(i) if deg0 < deg else ""
        say("Write output surface model..")
        if output:
            name = str(output) + suffix
            say(name)
            sr.save_deformed_surface(name)
        if deform:
            name = str(deform) + suffix
            say(name)
            sr.save_deformed_sphere(name)
        if output_property:
            name = str(output_property) + suffix
            say(name)
            sr.save_deformed_property(name)
        if bary:
            name = str(bary) + suffix
            say(name)
            sr.save_bary(name)


def resample(
    sphere,
    ref,
    values: Sequence[np.ndarray] | np.ndarray,
    coeff=None,
    *,
    nneighbor: bool = False,
    deg: int = -1,
) -> list[np.ndarray]:
    """Resample per-vertex ``values`` defined on ``sphere`` onto ``ref`` (float32 results).

    ``values`` is one 1-D array or a sequence of them (one value per ``sphere``
    vertex).  With ``coeff`` the SH deformation (full degree, or capped by ``deg``)
    is applied first.  Equivalent to ``SphericalRemesh -s sphere -r ref [-c coeff]
    -p ... --outputProperty ...`` without the file round trip.
    """
    arrs = [values] if isinstance(values, np.ndarray) and values.ndim == 1 else list(values)
    sr = SphericalRemeshing(
        None,
        sphere,
        coeff,
        False,
        ref,
        None,
        [(f"p{i}", a) for i, a in enumerate(arrs)],
        not nneighbor,
        deg,
        False,
    )
    sr.deform(sr.degree())
    return sr.deformed_data
