"""Read-only adapter for a FreeSurfer 6 subject directory (``surf/``, ``label/``).

Surfaces are in tkRAS; converting them needs the volume geometry of ``mri/orig.mgz``
(``pybiscuit.io.frames``). Nothing here needs a FreeSurfer installation: files are
read with NiBabel.
"""

from __future__ import annotations

from pathlib import Path

from ...config import Config
from ..model import InputError, InputFindings
from ._common import make_product

__all__ = ["REQUIRED", "find_freesurfer", "is_freesurfer_subject"]

_HEMIS = ("lh", "rh")
REQUIRED = tuple(f"{h}.{s}" for h in _HEMIS for s in ("white", "pial", "inflated", "sphere"))
_OPTIONAL_SURF = ("sphere.reg", "smoothwm", "orig")
_OPTIONAL_CURV = ("curv", "sulc", "thickness", "area")
_LABELS = ("aparc.annot", "aparc.a2009s.annot", "aparc.DKTatlas.annot", "aparc.DKTatlas40.annot")
_MRI = ("orig.mgz", "T1.mgz", "brain.mgz")


def is_freesurfer_subject(path: str | Path) -> bool:
    p = Path(path)
    return (p / "surf" / "lh.white").is_file() and (p / "surf" / "rh.white").is_file()


def find_freesurfer(path: str | Path, config: Config | None = None) -> InputFindings:
    root = Path(path).resolve()
    surf_dir = root / (config["fs_surf_model_path"] if config else "surf")
    label_dir = root / (config["fs_label_model_path"] if config else "label")
    if not surf_dir.is_dir():
        raise InputError(f"{root}: no {surf_dir.name}/ directory; not a FreeSurfer subject")
    f = InputFindings("freesurfer", str(root), "freesurfer", required=REQUIRED)
    f.source = {"layout": "FreeSurfer subject directory", "surf_dir": surf_dir.name}
    import nibabel.freesurfer as nfs

    for h in _HEMIS:
        for s in ("white", "pial", "inflated", "sphere", *_OPTIONAL_SURF):
            p = surf_dir / f"{h}.{s}"
            if not p.is_file():
                continue
            try:
                verts, faces, meta = nfs.read_geometry(str(p), read_metadata=True)
            except Exception as exc:  # NiBabel raises several types for bad files
                f.errors.append(f"{h}.{s}: cannot read surface ({exc})")
                continue
            info = {"n_vertices": int(len(verts)), "n_faces": int(len(faces))}
            if "cras" in meta:
                info["c_ras"] = [float(c) for c in meta["cras"]]
            make_product(f"{h}.{s}", p, "surface", "tkRAS", findings=f, **info)
        for c in _OPTIONAL_CURV:
            p = surf_dir / f"{h}.{c}"
            if p.is_file():
                make_product(f"{h}.{c}", p, "metric", None, findings=f)
        for lab in _LABELS:
            p = label_dir / f"{h}.{lab}"
            if p.is_file():
                make_product(f"{h}.{lab}", p, "label", None, findings=f)
    for m in _MRI:
        p = root / "mri" / m
        if p.is_file():
            make_product(f"mri/{m}", p, "volume", "scanner", findings=f)
    if "mri/orig.mgz" not in f.products:
        f.warnings.append(
            "mri/orig.mgz not found: tkRAS -> scanner RAS conversion needs its volume geometry"
        )
    for h in _HEMIS:
        a, b = f.products.get(f"{h}.white"), f.products.get(f"{h}.pial")
        if a and b and a.meta["n_vertices"] != b.meta["n_vertices"]:
            f.errors.append(f"{h}: white and pial vertex counts differ (no correspondence)")
    return f
