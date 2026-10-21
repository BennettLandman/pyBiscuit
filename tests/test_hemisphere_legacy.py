"""Hemisphere separation on the legacy MaCRUISE sample (B5.2). PROVISIONAL: no image oracle yet."""

import numpy as np
import pytest

from conftest import MIPAV
from pybiscuit.io.freesurfer import read_asc
from pybiscuit.mesh.topology import topology_summary
from pybiscuit.surfprep import frames
from pybiscuit.surfprep.hemisphere import separate_hemispheres

OUT = MIPAV / "sample-macruise" / "output"
pytestmark = pytest.mark.reference


@pytest.fixture(scope="module")
def sample():
    native = read_asc(OUT / "Surfaces_FreeView" / "target_image_GMimg_innerSurf.asc", frame="RAS")
    mni = read_asc(OUT / "Surfaces_MNI" / "target_image_GMimg_innerSurf.asc", frame="RAS")
    return native, mni


@pytest.fixture(scope="module")
def by_aff(sample):
    native, _ = sample
    aff = frames.read_aff(OUT / "Surfaces_MNI" / "aff.txt")
    ref = frames.reference_from_affine(native.vertices, aff)
    return separate_hemispheres(native, ref)


@pytest.fixture(scope="module")
def by_labels(sample):
    native, _ = sample
    ref = frames.reference_from_labels(native.vertices, OUT / "SegRefine" / "SEG_refine.nii.gz")
    return separate_hemispheres(native, ref)


def test_aff_maps_mni_to_native(sample):
    native, mni = sample
    aff = frames.read_aff(OUT / "Surfaces_MNI" / "aff.txt")
    back = frames.reference_from_affine(native.vertices, aff)
    assert np.abs(back - mni.vertices).max() < 1e-4


@pytest.mark.parametrize("which", ["by_aff", "by_labels"])
def test_both_hemispheres_closed_and_sided(which, request):
    r = request.getfixturevalue(which)
    for h in (r.lh, r.rh):
        t = topology_summary(h)
        assert t.is_closed and t.is_manifold and t.is_consistently_oriented
        assert t.genus == 0 and t.n_components == 1
    assert r.lh.vertices[:, 0].mean() < 0 < r.rh.vertices[:, 0].mean()
    d = r.diagnostics
    n_patch = sum(d.n_patch.values()) - sum(d.loop_sizes)
    assert (r.lh_mask == 0).sum() + (r.rh_mask == 0).sum() == n_patch
    # almost every cortex vertex survives; the rest is the bridge band and cut-off knobs
    assert r.hemi_map.shape[0] > 0.999 * 250874


def test_label_frame_matches_affine_frame(by_aff, by_labels):
    def side(r):
        a = np.full(250874, -1)
        a[r.hemi_map[:, 0]] = r.hemi_map[:, 2]
        return a

    a, b = side(by_aff), side(by_labels)
    both = (a >= 0) & (b >= 0)
    assert (a[both] == b[both]).mean() > 0.999


def test_label_midline_close_to_mni_axis():
    n = frames.midline_normal_from_seg(OUT / "SegRefine" / "SEG_refine.nii.gz")
    aff = frames.read_aff(OUT / "Surfaces_MNI" / "aff.txt")
    ax = aff[:3, 0] / np.linalg.norm(aff[:3, 0])
    assert np.degrees(np.arccos(n @ ax)) < 3.0
