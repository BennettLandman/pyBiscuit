"""Integration test for full MaCRUISE surface preparation pipeline (B5.1-B5.7)."""

import nibabel as nib
import numpy as np

from fixtures.bridged import bridged_ellipsoids
from pybiscuit.io import read_vtk
from pybiscuit.surfprep import run_macruise_surfprep


def test_run_macruise_surfprep_synthetic(tmp_path):
    mesh = bridged_ellipsoids(level=3)
    mesh_path = tmp_path / "inner_surface.vtk"
    from pybiscuit.surfprep.freesurfer import write_matlab_vtk

    write_matlab_vtk(mesh_path, mesh.vertices, mesh.faces)

    # Synthetic T1 and SEG_refine
    # 40x40x40 grid, 1mm spacing, center at 0
    t1_data = np.full((40, 40, 40), 100, dtype=np.float32)
    seg_data = np.zeros((40, 40, 40), dtype=np.int32)
    # Left hemisphere labels (x < 20)
    seg_data[5:18, 15:25, 15:25] = 45  # Left WM
    seg_data[3:5, 15:25, 15:25] = 101  # Left cortex
    # Right hemisphere labels (x > 20)
    seg_data[22:35, 15:25, 15:25] = 44  # Right WM
    seg_data[35:37, 15:25, 15:25] = 100  # Right cortex

    affine = np.eye(4)
    affine[:3, 3] = -20.0
    t1_img = nib.Nifti1Image(t1_data, affine)
    seg_img = nib.Nifti1Image(seg_data, affine)

    out_dir = tmp_path / "out"
    results = run_macruise_surfprep(
        inner_surface=mesh_path,
        seg_refine=seg_img,
        t1=t1_img,
        out_dir=out_dir,
        surf_smooth=0,
    )

    # Check that expected outputs were generated
    for h in ("lh", "rh"):
        for s in ("white", "gray", "mid", "smoothwm", "inflated", "sphere"):
            assert f"{h}.{s}" in results
            assert results[f"{h}.{s}"].is_file()
            m = read_vtk(results[f"{h}.{s}"])
            assert m.n_vertices > 0
            assert m.n_faces > 0

        for feat in ("thickness", "sulc", "mask"):
            assert f"{h}.{feat}" in results
            assert results[f"{h}.{feat}"].is_file()
            vals = np.loadtxt(results[f"{h}.{feat}"])
            assert len(vals) == read_vtk(results[f"{h}.white"]).n_vertices

    # Check auxiliary volumes
    for vol in ("brain.finalsurfs", "wm", "filled", "aseg"):
        assert vol in results
        assert results[vol].is_file()

    assert "hemi_map" in results
    assert results["hemi_map"].is_file()
