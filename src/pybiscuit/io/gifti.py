"""GIfTI surface I/O (NIFTI_INTENT_POINTSET + TRIANGLE, per-vertex scalars)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
from nibabel import gifti

from pybiscuit.mesh.model import Mesh


def read_gifti(path) -> Mesh:
    img = nib.load(str(path))
    verts = faces = None
    point_data: dict[str, np.ndarray] = {}
    for i, da in enumerate(img.darrays):
        intent = gifti.gifti.intent_codes.niistring[da.intent] if da.intent else ""
        if intent == "NIFTI_INTENT_POINTSET":
            verts = da.data
        elif intent == "NIFTI_INTENT_TRIANGLE":
            faces = da.data
        else:
            name = (da.meta or {}).get("Name") or f"array{i}"
            point_data[name] = np.asarray(da.data)
    if verts is None or faces is None:
        raise ValueError(f"{path}: missing POINTSET or TRIANGLE array")
    return Mesh(verts, faces, point_data, units="mm")


def write_gifti(path, mesh: Mesh) -> None:
    arrays = [
        gifti.GiftiDataArray(
            mesh.vertices.astype(np.float32),
            intent="NIFTI_INTENT_POINTSET",
            datatype="NIFTI_TYPE_FLOAT32",
        ),
        gifti.GiftiDataArray(
            mesh.faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE", datatype="NIFTI_TYPE_INT32"
        ),
    ]
    for name, arr in mesh.point_data.items():
        arr = np.asarray(arr)
        dt = np.float32 if arr.dtype.kind == "f" else np.int32
        arrays.append(
            gifti.GiftiDataArray(
                arr.astype(dt),
                intent="NIFTI_INTENT_NONE",
                datatype="NIFTI_TYPE_FLOAT32" if dt is np.float32 else "NIFTI_TYPE_INT32",
                meta=gifti.GiftiMetaData({"Name": name}),
            )
        )
    nib.save(gifti.GiftiImage(darrays=arrays), str(path))


# --- Standard exports (B12.1): *.surf.gii, *.func.gii, *.label.gii -------------------------

_STRUCTURE = {"lh": "CortexLeft", "rh": "CortexRight"}
# surface kind -> (AnatomicalStructureSecondary, GeometricType) as used by Connectome Workbench
_SURFACE_KINDS = {
    "white": ("GrayWhite", "Anatomical"),
    "pial": ("Pial", "Anatomical"),
    "mid": ("MidThickness", "Anatomical"),
    "inflated": (None, "Inflated"),
    "sphere": (None, "Sphere"),
    "sphere.reg": (None, "Sphere"),
}


def hemi_structure(hemi: str) -> str:
    """``"lh"``/``"left"`` -> ``CortexLeft``; ``"rh"``/``"right"`` -> ``CortexRight``."""
    key = hemi.lower()[0] + "h"
    if key not in _STRUCTURE:
        raise ValueError(f"hemisphere must be lh or rh, got {hemi!r}")
    return _STRUCTURE[key]


def _meta(**kv) -> gifti.GiftiMetaData:
    return gifti.GiftiMetaData({k: str(v) for k, v in kv.items() if v is not None})


def write_surf_gii(path, mesh: Mesh, hemi: str, kind: str = "white") -> None:
    """Write ``*.surf.gii`` with Workbench-style anatomical metadata.

    ``kind``: white, pial, mid, inflated, sphere or sphere.reg (sets
    ``AnatomicalStructureSecondary`` and ``GeometricType``). Vertices are RAS+ mm.
    """
    if kind not in _SURFACE_KINDS:
        raise ValueError(f"unknown surface kind {kind!r}; expected one of {sorted(_SURFACE_KINDS)}")
    secondary, geom = _SURFACE_KINDS[kind]
    structure = hemi_structure(hemi)
    cs = gifti.GiftiCoordSystem(
        dataspace="NIFTI_XFORM_SCANNER_ANAT", xformspace="NIFTI_XFORM_SCANNER_ANAT"
    )
    pts = gifti.GiftiDataArray(
        mesh.vertices.astype(np.float32),
        intent="NIFTI_INTENT_POINTSET",
        datatype="NIFTI_TYPE_FLOAT32",
        coordsys=cs,
        meta=_meta(
            AnatomicalStructurePrimary=structure,
            AnatomicalStructureSecondary=secondary,
            GeometricType=geom,
        ),
    )
    tri = gifti.GiftiDataArray(
        mesh.faces.astype(np.int32),
        intent="NIFTI_INTENT_TRIANGLE",
        datatype="NIFTI_TYPE_INT32",
        meta=_meta(AnatomicalStructurePrimary=structure),
    )
    img = gifti.GiftiImage(darrays=[pts, tri], meta=_meta(AnatomicalStructurePrimary=structure))
    nib.save(img, str(path))


def write_func_gii(path, data, hemi: str, names=None, intent: str = "NIFTI_INTENT_NONE") -> None:
    """Write ``*.func.gii``: ``data`` is (N,) or (N, K); one data array per column."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    names = list(names) if names is not None else [f"col{i}" for i in range(arr.shape[1])]
    if len(names) != arr.shape[1]:
        raise ValueError("names must match the number of columns")
    structure = hemi_structure(hemi)
    darrays = [
        gifti.GiftiDataArray(
            np.ascontiguousarray(arr[:, j]),
            intent=intent,
            datatype="NIFTI_TYPE_FLOAT32",
            meta=_meta(Name=names[j], AnatomicalStructurePrimary=structure),
        )
        for j in range(arr.shape[1])
    ]
    nib.save(
        gifti.GiftiImage(darrays=darrays, meta=_meta(AnatomicalStructurePrimary=structure)),
        str(path),
    )


def write_label_gii(path, labels, table, hemi: str, name: str = "label") -> None:
    """Write ``*.label.gii`` with ``table`` (a :class:`~pybiscuit.io.labels.LabelTable`) embedded.

    Every table entry is written verbatim (key, name, RGB). A key 0 present in ``labels`` but
    absent from the table is added as a transparent ``???`` background entry.
    """
    lab = np.asarray(labels).astype(np.int32)
    lt = gifti.GiftiLabelTable()
    for i in table.ids():
        r, g, b = table.rgb(i)
        lt.labels.append(
            gifti.GiftiLabel(key=i, red=r / 255, green=g / 255, blue=b / 255, alpha=1.0)
        )
        lt.labels[-1].label = table.name(i)
    if 0 not in table.entries and (lab == 0).any():
        bg = gifti.GiftiLabel(key=0, red=0.0, green=0.0, blue=0.0, alpha=0.0)
        bg.label = "???"
        lt.labels.insert(0, bg)
    structure = hemi_structure(hemi)
    da = gifti.GiftiDataArray(
        lab,
        intent="NIFTI_INTENT_LABEL",
        datatype="NIFTI_TYPE_INT32",
        meta=_meta(Name=name, AnatomicalStructurePrimary=structure),
    )
    img = gifti.GiftiImage(
        darrays=[da], labeltable=lt, meta=_meta(AnatomicalStructurePrimary=structure)
    )
    nib.save(img, str(path))


def read_label_gii(path):
    """Return ``(labels int32[N], LabelTable)`` from a ``*.label.gii``."""
    from .labels import LabelTable

    img = nib.load(str(path))
    entries = {
        lb.key: (lb.label, (round(lb.red * 255), round(lb.green * 255), round(lb.blue * 255)))
        for lb in img.labeltable.labels
    }
    return np.asarray(img.darrays[0].data, dtype=np.int32), LabelTable(entries)
