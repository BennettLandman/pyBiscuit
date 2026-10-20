from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pybiscuit.io.frames import (
    mesh_scanner_to_tkras,
    mesh_tkras_to_scanner,
    scanner_ras_to_tkras,
    tkras_to_scanner_ras,
    vox2ras_tkr,
)
from pybiscuit.io.freesurfer import read_asc
from pybiscuit.mesh.model import Mesh

MIPAV = Path.home() / "dev" / "mipav"  # same as tests/conftest.py
SAMPLE = MIPAV / "sample-macruise" / "output"


def _img():
    # 1 x 1 x 1.5 mm, shape (10, 20, 30), x flipped, y/z permuted, offset
    aff = np.array([[-1.0, 0, 0, 5.0], [0, 0, 1.5, -7.0], [0, 1.0, 0, 11.0], [0, 0, 0, 1]])
    return nib.Nifti1Image(np.zeros((10, 20, 30), np.float32), aff)


def test_vox2ras_tkr_matches_nibabel_mgh():
    img = _img()
    from nibabel.freesurfer.mghformat import MGHHeader

    h = MGHHeader()
    h.set_data_shape((10, 20, 30))
    h.set_zooms((1.0, 1.0, 1.5))
    np.testing.assert_allclose(vox2ras_tkr(img), h.get_vox2ras_tkr())


def test_known_voxel():
    img = _img()
    # tkRAS of voxel (i,j,k): tkr @ [i,j,k,1]; scanner RAS: affine @ [i,j,k,1]
    for ijk in [(0, 0, 0), (3, 7, 12), (9, 19, 29)]:
        h = np.array([*ijk, 1.0])
        tk = (vox2ras_tkr(img) @ h)[:3]
        got = tkras_to_scanner_ras(tk[None], img)[0]
        np.testing.assert_allclose(got, (img.affine @ h)[:3], atol=1e-9)
    # voxel centre (5, 10, 15) is tkRAS origin
    np.testing.assert_allclose(vox2ras_tkr(img) @ [5, 10, 15, 1], [0, 0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(
        tkras_to_scanner_ras([[0, 0, 0]], img)[0], (img.affine @ [5, 10, 15, 1])[:3], atol=1e-9
    )


def test_roundtrip_and_header_input():
    img = _img()
    pts = np.random.default_rng(0).normal(size=(50, 3)) * 20
    s = tkras_to_scanner_ras(pts, img.header)
    np.testing.assert_allclose(s, tkras_to_scanner_ras(pts, img))
    np.testing.assert_allclose(scanner_ras_to_tkras(s, img), pts, atol=1e-9)


def test_mesh_helpers():
    img = _img()
    m = Mesh(np.eye(3) * 4, np.array([[0, 1, 2]]), {"a": np.arange(3.0)}, frame="tkRAS")
    r = mesh_tkras_to_scanner(m, img)
    assert r.frame == "RAS" and m.frame == "tkRAS"
    back = mesh_scanner_to_tkras(r, img)
    assert back.frame == "tkRAS"
    np.testing.assert_allclose(back.vertices, m.vertices, atol=1e-9)
    with pytest.raises(ValueError):
        mesh_tkras_to_scanner(r, img)


@pytest.mark.reference
def test_legacy_sample_mask_diagnostic(capsys):
    from scipy import ndimage

    t1 = nib.load(SAMPLE / "Raw" / "T1.nii.gz")
    seg = nib.load(SAMPLE / "SegRefine" / "SEG_refine.nii.gz")
    mask = np.asarray(seg.dataobj) != 0
    inv = np.linalg.inv(seg.affine)

    def frac(pts, r):
        m = ndimage.binary_dilation(mask, iterations=r) if r else mask
        ijk = np.rint(pts @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < mask.shape), axis=1)
        out = np.zeros(len(pts), bool)
        out[ok] = m[tuple(ijk[ok].T)]
        return out.mean()

    lines = [
        f"T1 shape {t1.shape}, SEG shape {seg.shape}, same affine: "
        f"{np.allclose(t1.affine, seg.affine)}"
    ]
    lines.append(
        f"T1 extent min {(t1.affine @ [0, 0, 0, 1])[:3]} (voxel 0) / max "
        f"{(t1.affine @ [*[s - 1 for s in t1.shape], 1])[:3]}"
    )
    fv = read_asc(SAMPLE / "Surfaces_FreeView" / "target_image_GMimg_innerSurf.asc")
    mni = read_asc(SAMPLE / "Surfaces_MNI" / "target_image_GMimg_innerSurf.asc")
    conv = tkras_to_scanner_ras(fv.vertices, t1)
    res = {}
    for name, pts in [
        ("FreeView converted", conv),
        ("FreeView raw", fv.vertices),
        ("MNI raw", mni.vertices),
    ]:
        res[name] = [frac(pts, r) for r in (0, 1, 2, 3)]
        lines.append(
            f"{name}: bbox min {pts.min(0).round(1)} max {pts.max(0).round(1)}; "
            f"inside mask r=0..3: {np.round(res[name], 4)}"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    # Loose sanity only. Finding: the legacy FreeView .asc is NOT in tkRAS; it already
    # lies in scanner RAS, so the unconverted control sits inside the mask and the
    # tkRAS->scanner conversion moves it out. Record, don't "fix": Q4 owns the semantics.
    assert res["FreeView raw"][3] > 0.95
    assert res["FreeView raw"][0] > res["FreeView converted"][0]
