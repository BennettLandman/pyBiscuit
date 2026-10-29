"""FreeSurfer-route input stage (brief 22): synthetic checks, byte-equality with image files."""

from pathlib import Path

import numpy as np
import pytest

from _open_examples import open_examples_root, subject_dir
from pybiscuit.io import read_vtk
from pybiscuit.surfprep.freesurfer import (
    format_f,
    mask_from_label,
    mid_vertices,
    prepare_fs_subject,
    read_cortex_label,
    vtk_text,
)

OPEN = open_examples_root()
SUBJECTS = ("0050952", "0050961", "0050971")


def test_format_f_six_decimals():
    assert format_f([1.5, -0.0000004, 2.0000006]) == "1.500000\n-0.000000\n2.000001\n"


def test_vtk_text_layout_and_roundtrip(tmp_path):
    v = np.array([[0, 0, 0], [1.25, 0, 0], [0, 1, 0.1234567]])
    f = np.array([[0, 1, 2]])
    txt = vtk_text(v, f)
    assert txt.startswith(
        "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\nPOINTS 3 float\n"
    )
    assert txt.endswith("POLYGONS 1 4\n3 0 1 2\n")
    p = tmp_path / "a.vtk"
    p.write_text(txt)
    m = read_vtk(p)
    assert m.faces.tolist() == [[0, 1, 2]]
    assert m.vertices[2, 2] == pytest.approx(0.123457)


def test_mid_uses_printed_vertices():
    # 0.0000004 prints as 0.000000, 0.0000006 as 0.000001: the mean is of printed values
    wm = np.array([[0.0000004, 0, 0]])
    gm = np.array([[0.0000006, 0, 0]])
    assert mid_vertices(wm, gm)[0, 0] == pytest.approx(0.0000005)


def test_label_mask(tmp_path):
    p = tmp_path / "lh.cortex.label"
    p.write_text("#!ascii label\n3\n0  1.0 2.0 3.0 0.0\n2  1.0 2.0 3.0 0.0\n4 1 2 3 0\n")
    idx = read_cortex_label(p)
    assert idx.tolist() == [0, 2, 4]
    assert mask_from_label(idx, 6).tolist() == [1, 0, 1, 0, 1, 0]


def _same(a: Path, b: Path) -> bool:
    return a.read_bytes() == b.read_bytes()


@pytest.mark.reference
@pytest.mark.parametrize("subject", SUBJECTS)
def test_matches_image_inputs(subject, tmp_path):
    """Our VTKs, mask and thickness are byte-identical to the files the image wrote."""
    if OPEN is None:
        pytest.skip("open-examples not available")
    src = subject_dir(OPEN, subject, tmp_path / "data")
    prepare_fs_subject(src, tmp_path)
    for h in ("lh", "rh"):
        for name in ("white", "gray", "mid", "inflated", "sphere"):
            assert _same(tmp_path / f"{h}.{name}.vtk", src / "surf" / f"{h}.{name}.vtk"), (h, name)
        assert _same(tmp_path / f"{h}.mask.txt", src / "roi" / f"{h}.mask.txt")
        assert _same(tmp_path / f"{h}.thickness.txt", src / "quant" / f"{h}.thickness.txt")
