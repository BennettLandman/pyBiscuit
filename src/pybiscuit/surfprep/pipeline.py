"""Full MaCRUISE surface preparation pipeline (plan B5.1-B5.7; spec §4).

Chains:
1. Hemisphere separation and medial-wall patching (spec §4.2, B5.2);
2. Volume preparation and synthesis: brain.finalsurfs, wm, filled, aseg (spec §4.3, §4.5, B5.3);
3. White refinement, pial deformation along outer level set, and thickness (spec §4.6, B5.4);
4. Mid surface calculation (spec §4.8, B5.5);
5. Inflation: smoothwm, inflated, sulc (spec §4.7, B5.6);
6. Spherical mapping: sphere.vtk (spec §4.7, B5.7).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from pybiscuit.io import read_asc, read_vtk
from pybiscuit.mesh.model import Mesh
from pybiscuit.surfprep.frames import (
    midline_normal_from_seg,
    reference_from_affine,
    rotation_to_x,
)
from pybiscuit.surfprep.freesurfer import (
    write_feature_txt,
    write_mask_txt,
    write_matlab_vtk,
)
from pybiscuit.surfprep.hemisphere import separate_hemispheres
from pybiscuit.surfprep.surface import prepare_hemisphere_surfaces
from pybiscuit.surfprep.volume import prepare_auxiliary_volumes

__all__ = ["run_macruise_surfprep"]


def _load_mesh(path: str | Path) -> Mesh:
    p = Path(path)
    if p.suffix.lower() == ".asc":
        return read_asc(p, frame="RAS")
    return read_vtk(p)


def run_macruise_surfprep(
    inner_surface: str | Path | Mesh,
    seg_refine: str | Path | nib.spatialimages.SpatialImage,
    t1: str | Path | nib.spatialimages.SpatialImage,
    out_dir: str | Path,
    *,
    surf_smooth: int = 0,
    outer_levelset: str | Path | nib.spatialimages.SpatialImage | None = None,
    aff_matrix: np.ndarray | None = None,
    mni_surface: str | Path | Mesh | None = None,
) -> dict[str, Path]:
    """Execute complete surface prep pipeline on MaCRUISE inputs.

    Parameters
    ----------
    inner_surface : Mesh or path to VTK/ASC inner surface
    seg_refine : nibabel image or path to SEG_refine.nii.gz
    t1 : nibabel image or path to T1.nii.gz
    out_dir : directory to write outputs
    surf_smooth : Laplacian smoothing iterations for white surface (default 0)
    outer_levelset : optional path or image of PyMaCRUISE outer level set
    aff_matrix : optional 4x4 matrix from aff.txt (MNI -> native)
    mni_surface : optional MNI-space inner surface

    Returns
    -------
    dict[str, Path] : paths to all generated outputs
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mri_dir = out_dir / "mri"
    mri_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load inputs
    inner_mesh = inner_surface if isinstance(inner_surface, Mesh) else _load_mesh(inner_surface)
    seg_img = seg_refine if hasattr(seg_refine, "affine") else nib.load(str(seg_refine))
    t1_img = t1 if hasattr(t1, "affine") else nib.load(str(t1))

    outer_ls_img = None
    if outer_levelset is not None:
        outer_ls_img = (
            outer_levelset if hasattr(outer_levelset, "affine") else nib.load(str(outer_levelset))
        )

    # 2. Reference frame for cutting plane
    if mni_surface is not None:
        mni_mesh = mni_surface if isinstance(mni_surface, Mesh) else _load_mesh(mni_surface)
        ref_verts = mni_mesh.vertices
    elif aff_matrix is not None:
        ref_verts = reference_from_affine(inner_mesh.vertices, aff_matrix)
    else:
        # Fit midline from SEG_refine
        normal = midline_normal_from_seg(seg_img)
        rot = rotation_to_x(normal)
        ref_verts = inner_mesh.vertices @ rot.T

    # 3. Hemisphere separation
    sep = separate_hemispheres(inner_mesh, ref_verts)

    results: dict[str, Path] = {}

    # Write hemi_map
    hemi_map_path = out_dir / "hemi_map.txt"
    np.savetxt(hemi_map_path, sep.hemi_map, fmt="%d")
    results["hemi_map"] = hemi_map_path

    # 4. Prepare each hemisphere
    hemis = {"lh": (sep.lh, sep.lh_mask), "rh": (sep.rh, sep.rh_mask)}

    for h, (mesh_h, mask_h) in hemis.items():
        # Mask
        mask_path = out_dir / f"{h}.mask.txt"
        write_mask_txt(mask_h, mask_path)
        results[f"{h}.mask"] = mask_path

        # Prep surfaces
        prep = prepare_hemisphere_surfaces(
            mesh_h,
            surf_smooth=surf_smooth,
            outer_levelset=outer_ls_img,
            mask=mask_h,
        )

        # Surfaces to write
        surfs = {
            "white": prep.white,
            "gray": prep.pial,
            "mid": prep.mid,
            "smoothwm": prep.smoothwm,
            "inflated": prep.inflated,
            "sphere": prep.sphere,
        }

        for sname, smesh in surfs.items():
            p = out_dir / f"{h}.{sname}.vtk"
            write_matlab_vtk(p, smesh.vertices, smesh.faces)
            results[f"{h}.{sname}"] = p

        # Thickness & sulc features
        thick_path = out_dir / f"{h}.thickness.txt"
        write_feature_txt(prep.thickness, thick_path)
        results[f"{h}.thickness"] = thick_path

        sulc_path = out_dir / f"{h}.sulc.txt"
        write_feature_txt(prep.sulc, sulc_path)
        results[f"{h}.sulc"] = sulc_path

    # 5. Volume preparation
    vol_results = prepare_auxiliary_volumes(t1_img, seg_img, mri_dir, conform=True)
    for vname, vpath in vol_results.items():
        results[vname] = vpath

    return results
