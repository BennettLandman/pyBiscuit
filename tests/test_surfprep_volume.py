"""Unit tests for volume synthesis and preparation (plan B5.3, spec §4.3, §4.5)."""

import nibabel as nib
import numpy as np

from pybiscuit.surfprep.volume import (
    FILLED_LEFT_LABELS,
    FILLED_RIGHT_LABELS,
    prepare_auxiliary_volumes,
    synthesize_aseg,
    synthesize_brain_finalsurfs,
    synthesize_filled,
    synthesize_wm,
)


def test_synthesize_aseg_cortex_and_subcortical():
    # Construct synthetic segmentation
    seg = np.zeros((10, 10, 10), dtype=np.int32)
    # Left cortex (odd >= 100)
    seg[1, 1, 1] = 101
    seg[1, 1, 2] = 103
    # Right cortex (even >= 100)
    seg[2, 2, 1] = 100
    seg[2, 2, 2] = 104
    # Specific mappings
    seg[3, 3, 3] = 45  # Left-Cerebral-White-Matter -> 2
    seg[4, 4, 4] = 44  # Right-Cerebral-White-Matter -> 41
    seg[5, 5, 5] = 35  # Brain-Stem -> 16
    seg[6, 6, 6] = 58  # Left-Putamen -> 12
    seg[7, 7, 7] = 57  # Right-Putamen -> 51
    seg[8, 8, 8] = 11  # 4th-Ventricle -> 15

    aseg = synthesize_aseg(seg)

    assert aseg[1, 1, 1] == 3
    assert aseg[1, 1, 2] == 3
    assert aseg[2, 2, 1] == 42
    assert aseg[2, 2, 2] == 42
    assert aseg[3, 3, 3] == 2
    assert aseg[4, 4, 4] == 41
    assert aseg[5, 5, 5] == 16
    assert aseg[6, 6, 6] == 12
    assert aseg[7, 7, 7] == 51
    assert aseg[8, 8, 8] == 15
    assert aseg[0, 0, 0] == 0


def test_synthesize_filled():
    seg = np.zeros((10, 10, 10), dtype=np.int32)
    for i, lbl in enumerate(FILLED_LEFT_LABELS):
        seg[i, 0, 0] = lbl
    for i, lbl in enumerate(FILLED_RIGHT_LABELS):
        seg[i, 1, 0] = lbl

    filled = synthesize_filled(seg)

    for i in range(len(FILLED_LEFT_LABELS)):
        assert filled[i, 0, 0] == 255
    for i in range(len(FILLED_RIGHT_LABELS)):
        assert filled[i, 1, 0] == 127
    assert filled[9, 9, 9] == 0


def test_synthesize_wm():
    t1 = np.full((10, 10, 10), 110, dtype=np.float32)
    seg = np.zeros((10, 10, 10), dtype=np.int32)

    # Kept as T1 (44, 45, 35)
    seg[0, 0, 0] = 44
    seg[0, 0, 1] = 45
    seg[0, 0, 2] = 35

    # Fixed to 250
    seg[1, 1, 0] = 36
    seg[1, 1, 1] = 51

    # Dropped to 0 (cortex or background)
    seg[2, 2, 0] = 101

    wm = synthesize_wm(t1, seg)

    assert wm[0, 0, 0] == 110
    assert wm[0, 0, 1] == 110
    assert wm[0, 0, 2] == 110
    assert wm[1, 1, 0] == 250
    assert wm[1, 1, 1] == 250
    assert wm[2, 2, 0] == 0
    assert wm[9, 9, 9] == 0


def test_synthesize_brain_finalsurfs():
    t1 = np.full((10, 10, 10), 85, dtype=np.float32)
    seg = np.zeros((10, 10, 10), dtype=np.int32)
    seg[2:5, 2:5, 2:5] = 101

    bfs = synthesize_brain_finalsurfs(t1, seg)

    assert (bfs[2:5, 2:5, 2:5] == 85).all()
    assert (bfs[0:2, 0:2, 0:2] == 0).all()


def test_prepare_auxiliary_volumes_end_to_end(tmp_path):
    t1_data = np.random.default_rng(42).uniform(0, 150, (30, 30, 30)).astype(np.float32)
    seg_data = np.zeros((30, 30, 30), dtype=np.int32)
    seg_data[10:20, 10:20, 10:20] = 45  # WM
    seg_data[5:10, 5:10, 5:10] = 101  # Left cortex

    affine = np.diag([1.5, 1.5, 1.5, 1.0])
    t1_img = nib.Nifti1Image(t1_data, affine)
    seg_img = nib.Nifti1Image(seg_data, affine)

    out_dir = tmp_path / "mri"
    results = prepare_auxiliary_volumes(t1_img, seg_img, out_dir, conform=False)

    for expected in ("brain.finalsurfs", "wm", "filled", "aseg"):
        assert expected in results
        assert results[expected].is_file()
        loaded = nib.load(results[expected])
        assert loaded.shape == (30, 30, 30)
