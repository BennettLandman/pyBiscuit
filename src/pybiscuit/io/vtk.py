"""VTK legacy POLYDATA reader/writer (ASCII and big-endian binary, triangles only).

Header convention (PyMaCRUISE): the title line may read
``PyMaCRUISE coordinate_frame=RAS units=mm``; these set ``Mesh.frame``/``Mesh.units``
and are written back when set.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np

from pybiscuit.mesh import Mesh

_VTK_TYPES = {
    "bit": "u1", "char": "i1", "signed_char": "i1", "unsigned_char": "u1",
    "short": "i2", "unsigned_short": "u2", "int": "i4", "unsigned_int": "u4",
    "long": "i8", "unsigned_long": "u8", "vtktypeint64": "i8", "vtktypeuint64": "u8",
    "vtktypeint32": "i4", "vtktypeuint32": "u4",
    "float": "f4", "double": "f8",
}  # fmt: skip
# first non-numeric character ends an ASCII numeric block ('e'/'E' allowed in exponents)
_NONNUM = re.compile(rb"(?<![0-9.])[A-Za-z]|[A-DF-Za-df-z]")
_FRAME = re.compile(r"coordinate_frame=(\S+)")
_UNITS = re.compile(r"units=(\S+)")


class _Reader:
    def __init__(self, data: bytes, path) -> None:
        self.d = data
        self.pos = 0
        self.path = path
        self.binary = False

    def err(self, msg: str) -> ValueError:
        return ValueError(f"{self.path}: {msg}")

    def line(self) -> str:
        end = self.d.find(b"\n", self.pos)
        if end < 0:
            end = len(self.d)
        s = self.d[self.pos : end].decode("latin-1").strip()
        self.pos = min(end + 1, len(self.d))
        return s

    def skip_ws(self) -> None:
        n = len(self.d)
        p = self.pos
        while p < n and self.d[p] in b" \t\r\n":
            p += 1
        self.pos = p

    def keyword_line(self) -> list[str] | None:
        self.skip_ws()
        if self.pos >= len(self.d):
            return None
        return self.line().split()

    def numbers(self, count: int, ntype: str) -> np.ndarray:
        """Read ``count`` values of VTK type ``ntype`` (ASCII or binary)."""
        if ntype not in _VTK_TYPES:
            raise self.err(f"unsupported data type {ntype!r}")
        code = _VTK_TYPES[ntype]
        if self.binary:
            dt = np.dtype(">" + code)
            nbytes = count * dt.itemsize
            buf = self.d[self.pos : self.pos + nbytes]
            if len(buf) != nbytes:
                raise self.err("truncated binary data")
            self.pos += nbytes
            return np.frombuffer(buf, dtype=dt).astype(dt.newbyteorder("="))
        m = _NONNUM.search(self.d, self.pos)
        end = m.start() if m else len(self.d)
        arr = np.fromstring(self.d[self.pos : end], sep=" ", dtype=np.float64)  # noqa: NPY201
        if arr.size < count:
            raise self.err(f"expected {count} values, found {arr.size}")
        if arr.size > count:
            raise self.err(f"found {arr.size} values, expected {count}")
        self.pos = end
        return arr.astype(np.dtype(code))


def read_vtk(path) -> Mesh:
    """Read a legacy VTK POLYDATA triangle mesh (ASCII or binary, optionally gzip-compressed)."""
    path = Path(path)
    data = path.read_bytes()
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    r = _Reader(data, path)
    version = r.line()
    if not version.startswith("# vtk DataFile"):
        raise r.err("not a legacy VTK file (missing '# vtk DataFile' header)")
    title = r.line()
    fmt = r.line().upper()
    if fmt not in ("ASCII", "BINARY"):
        raise r.err(f"unknown VTK format {fmt!r}")
    r.binary = fmt == "BINARY"
    ds = r.keyword_line()
    if not ds or ds[0].upper() != "DATASET" or ds[1].upper() != "POLYDATA":
        raise r.err(f"only DATASET POLYDATA is supported, got {' '.join(ds or [])!r}")

    verts = faces = None
    point_data: dict[str, np.ndarray] = {}
    meta: dict = {"title": title, "vtk_version": version.split()[-1], "vtk_ascii": not r.binary}
    n_points = 0
    in_point_data = False
    while (tok := r.keyword_line()) is not None:
        key = tok[0].upper()
        if key == "POINTS":
            n_points = int(tok[1])
            meta["points_type"] = tok[2].lower()
            verts = r.numbers(n_points * 3, tok[2].lower()).reshape(n_points, 3)
        elif key == "POLYGONS":
            m, size = int(tok[1]), int(tok[2])
            arr = r.numbers(size, "int")
            if size != 4 * m or (m and not np.all(arr[0::4] == 3)):
                raise r.err("POLYGONS must contain triangles only")
            faces = arr.reshape(m, 4)[:, 1:]
        elif key in ("VERTICES", "LINES", "TRIANGLE_STRIPS"):
            raise r.err(f"unsupported POLYDATA section {key}")
        elif key == "POINT_DATA":
            in_point_data = True
            if verts is not None and int(tok[1]) != n_points:
                raise r.err("POINT_DATA count differs from POINTS count")
        elif key == "CELL_DATA":
            break  # per-face data is not modelled; ignore the remainder
        elif key in ("SCALARS", "VECTORS", "NORMALS") and in_point_data:
            name, ntype = tok[1], tok[2].lower()
            ncomp = 3 if key != "SCALARS" else (int(tok[3]) if len(tok) > 3 else 1)
            if key == "SCALARS":
                look = r.keyword_line()
                if not look or look[0].upper() != "LOOKUP_TABLE":
                    raise r.err(f"SCALARS {name}: missing LOOKUP_TABLE line")
            vals = r.numbers(n_points * ncomp, ntype)
            point_data[name] = vals if ncomp == 1 else vals.reshape(n_points, ncomp)
        elif key == "FIELD":
            break  # field arrays are not modelled
        else:
            raise r.err(f"unexpected keyword {tok[0]!r}")
    if verts is None or faces is None:
        raise r.err("missing POINTS or POLYGONS section")
    if faces.size and (faces.min() < 0 or faces.max() >= n_points):
        raise r.err("face index out of range")

    frame = units = None
    if (mf := _FRAME.search(title)) is not None:
        frame = mf.group(1)
    if (mu := _UNITS.search(title)) is not None:
        units = mu.group(1)
    return Mesh(verts, faces, point_data=point_data, frame=frame, units=units, meta=meta)


def _vtk_type(a: np.ndarray) -> tuple[str, np.ndarray]:
    if a.dtype == np.float32:
        return "float", a
    if a.dtype.kind == "f":
        return "double", a
    if a.dtype.kind in "iub":
        if a.dtype.kind == "u" and a.dtype.itemsize == 1:
            return "unsigned_char", a
        return "int", a.astype(np.int32)
    raise ValueError(f"unsupported point_data dtype {a.dtype}")


def _fmt_block(a: np.ndarray, vtype: str) -> bytes:
    if vtype == "float":
        fmt = "%.9g"
    elif vtype == "double":
        fmt = "%.17g"
    else:
        fmt = "%d"
    a = np.atleast_2d(a.reshape(a.shape[0], -1))
    row = " ".join([fmt] * a.shape[1])
    return ("\n".join(row % tuple(x) for x in a.tolist()) + "\n").encode()


def write_vtk(mesh: Mesh, path, binary: bool = False, dtype: str | None = None) -> None:
    """Write ``mesh`` as legacy VTK POLYDATA.

    ``dtype`` is the POINTS type, ``"float"`` or ``"double"`` (default: the type the
    mesh was read with, else double). Binary output is big-endian per the VTK spec.
    """
    ptype = dtype or mesh.meta.get("points_type", "double")
    if ptype not in ("float", "double"):
        raise ValueError("dtype must be 'float' or 'double'")
    parts = []
    if mesh.frame or mesh.units:
        parts.append("PyMaCRUISE")
        if mesh.frame:
            parts.append(f"coordinate_frame={mesh.frame}")
        if mesh.units:
            parts.append(f"units={mesh.units}")
        title = " ".join(parts)
    else:
        title = str(mesh.meta.get("title", "vtk output")).replace("\n", " ")
    n, m = mesh.n_vertices, mesh.n_faces
    out = [f"# vtk DataFile Version 3.0\n{title}\n{'BINARY' if binary else 'ASCII'}\n"
           f"DATASET POLYDATA\nPOINTS {n} {ptype}\n".encode()]  # fmt: skip
    cells = np.empty((m, 4), dtype=np.int32)
    cells[:, 0] = 3
    cells[:, 1:] = mesh.faces
    if binary:
        out.append(mesh.vertices.astype(">f4" if ptype == "float" else ">f8").tobytes())
        out.append(f"\nPOLYGONS {m} {4 * m}\n".encode())
        out.append(cells.astype(">i4").tobytes())
        out.append(b"\n")
    else:
        out.append(_fmt_block(mesh.vertices, ptype))
        out.append(f"POLYGONS {m} {4 * m}\n".encode())
        out.append(_fmt_block(cells, "int"))
    if mesh.point_data:
        out.append(f"POINT_DATA {n}\n".encode())
        for name, arr in mesh.point_data.items():
            a = np.asarray(arr)
            if a.shape[0] != n:
                raise ValueError(f"point_data[{name!r}] length {a.shape[0]} != {n} vertices")
            vtype, a = _vtk_type(a)
            safe = re.sub(r"\s+", "_", name) or "data"
            ncomp = 1 if a.ndim == 1 else int(np.prod(a.shape[1:]))
            if ncomp == 3 and a.ndim == 2:
                out.append(f"VECTORS {safe} {vtype}\n".encode())
            else:
                out.append(f"SCALARS {safe} {vtype} {ncomp}\nLOOKUP_TABLE default\n".encode())
            if binary:
                code = _VTK_TYPES[vtype]
                out.append(np.ascontiguousarray(a).astype(">" + code).tobytes() + b"\n")
            else:
                out.append(_fmt_block(a, vtype))
    Path(path).write_bytes(b"".join(out))
