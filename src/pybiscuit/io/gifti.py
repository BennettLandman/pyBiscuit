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
