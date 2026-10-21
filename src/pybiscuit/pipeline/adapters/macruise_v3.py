"""Adapter for a legacy MaCRUISE v3 output directory (what ``biscuit_FC_v2.2.sif
--surf_recon macruise`` consumes; see docs/io-contract.md).

``Surfaces_FreeView/*.asc`` are FreeSurfer-ASCII meshes that are already in
**scanner RAS** (not tkRAS); ``Surfaces_MNI/*.asc`` are the same meshes moved by the
rigid transform in ``Surfaces_MNI/aff.txt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...config import Config
from ..model import InputError, InputFindings
from ._common import check_same_grid, make_product

__all__ = [
    "REQUIRED",
    "apply_affine",
    "find_macruise_v3",
    "is_macruise_v3",
    "locate_root",
    "native_to_mni_matrix",
    "read_mni_transform",
]

REQUIRED = ("inner_surface", "inner_surface_mni", "mni_transform", "seg_refine", "t1")
_PREFIX = "target_image_GMimg_"
_GRID_PRODUCTS = ["t1", "seg_refine"]


def locate_root(path: str | Path) -> Path | None:
    """The MaCRUISE output directory (the one holding ``Surfaces_FreeView``), given it
    or a parent such as the subject directory."""
    p = Path(path)
    for cand in (p, *(p / n for n in ("Output", "output", "outputs"))):
        if (cand / "Surfaces_FreeView").is_dir():
            return cand
    return None


def is_macruise_v3(path: str | Path) -> bool:
    return locate_root(path) is not None


def read_mni_transform(path: str | Path, *, rigid_tol: float = 1e-4) -> np.ndarray:
    """Parse ``aff.txt`` (4x4, row-major text) and require a proper rigid transform."""
    m = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if m.shape != (4, 4):
        raise InputError(f"{path}: expected a 4x4 matrix, got shape {m.shape}")
    if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-9):
        raise InputError(f"{path}: last row must be 0 0 0 1, got {m[3]}")
    r = m[:3, :3]
    if np.abs(r @ r.T - np.eye(3)).max() > rigid_tol or np.linalg.det(r) <= 0:
        raise InputError(
            f"{path}: upper 3x3 block is not a proper rotation "
            f"(|R R^T - I| max {np.abs(r @ r.T - np.eye(3)).max():.2g}, det {np.linalg.det(r):.4f})"
        )
    return m


def native_to_mni_matrix(aff: np.ndarray) -> np.ndarray:
    """Transform taking Surfaces_FreeView (native RAS) vertices onto the Surfaces_MNI ones.

    ``aff.txt`` itself maps MNI -> native; measured on the sample, the inverse reproduces
    the MNI innerSurf vertices to 5e-6 mm (the files carry 5 decimals).
    """
    return np.linalg.inv(aff)


def apply_affine(m: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (N, 3) points."""
    v = np.asarray(vertices, dtype=np.float64)
    return v @ m[:3, :3].T + m[:3, 3]


def _asc_counts(path: Path) -> tuple[int, int]:
    with open(path) as fh:
        first = fh.readline()
        if not first.startswith("#!ascii"):
            raise InputError(f"{path}: not a FreeSurfer ASCII surface (no '#!ascii' header)")
        nv, nf = (int(t) for t in fh.readline().split()[:2])
    return nv, nf


def find_macruise_v3(path: str | Path, config: Config | None = None) -> InputFindings:
    out = locate_root(path)
    if out is None:
        raise InputError(
            f"{path}: no MaCRUISE v3 output found (expected a Surfaces_FreeView/ directory in it "
            "or in Output/, output/, outputs/)"
        )
    out = out.resolve()
    f = InputFindings("macruise-v3", str(out), "macruise", required=REQUIRED)
    f.source = {"layout": "MaCRUISE v3 output tree"}
    counts: dict[str, tuple[int, int]] = {}

    def surf(name: str, rel: str, frame: str) -> None:
        p = out / rel
        if not p.is_file():
            return
        try:
            nv, nf = _asc_counts(p)
        except (InputError, ValueError) as exc:
            f.errors.append(f"{name}: {exc}")
            return
        counts[name] = (nv, nf)
        make_product(name, p, "surface", frame, findings=f, n_vertices=nv, n_faces=nf,
                     asc_frame_note="scanner RAS as written by MaCRUISE (read with frame='RAS')"
                     if frame == "RAS" else "MNI space; rigid transform of the native mesh")

    for which in ("inner", "central", "outer"):
        surf(f"{which}_surface", f"Surfaces_FreeView/{_PREFIX}{which}Surf.asc", "RAS")
        surf(f"{which}_surface_mni", f"Surfaces_MNI/{_PREFIX}{which}Surf.asc", "MNI")
        vol = out / "Surfaces" / f"{_PREFIX}{which}Vol.nii.gz"
        if vol.is_file():
            make_product(f"{which}_levelset", vol, "volume", "RAS", findings=f)
    for which in ("inner", "central", "outer"):
        a, b = counts.get(f"{which}_surface"), counts.get(f"{which}_surface_mni")
        if a and b and a != b:
            f.errors.append(f"{which}: native {a} and MNI {b} vertex/face counts differ")

    aff = out / "Surfaces_MNI" / "aff.txt"
    if aff.is_file():
        try:
            m = read_mni_transform(aff)
        except InputError as exc:
            f.errors.append(str(exc))
        else:
            make_product("mni_transform", aff, "transform", None, findings=f, matrix=m.tolist(),
                         direction="aff.txt maps MNI -> native RAS; "
                         "its inverse maps native onto the MNI meshes")
    for name, rel in (
        ("seg_refine", "SegRefine/SEG_refine.nii.gz"),
        ("t1", "Raw/T1.nii.gz"),
        ("seg_refine_mni", "Surfaces_MNI/SEG_refine_MNI.nii.gz"),
        ("t1_mni", "Surfaces_MNI/T1_N4_MNI.nii.gz"),
    ):
        p = out / rel
        if p.is_file():
            make_product(name, p, "volume", "MNI" if name.endswith("_mni") else "RAS", findings=f)
    if "t1" in f.products and "seg_refine" in f.products:
        # MaCRUISE writes in the T1 header's orientation; RAS is not guaranteed, so only
        # require that T1 and SEG_refine share a grid
        check_same_grid(f, _GRID_PRODUCTS, require_ras=False)
    return f
