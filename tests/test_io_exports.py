import json

import nibabel as nib
import numpy as np
import pytest

from conftest import OPT
from pybiscuit.io import (
    read_gifti,
    read_label_gii,
    read_label_table,
    read_protocol_table,
    read_stats_tsv,
    write_func_gii,
    write_label_gii,
    write_stats_tsv,
    write_surf_gii,
)
from pybiscuit.io.labels import hsv_fallback_colors
from pybiscuit.mesh.model import Mesh

TETRA_V = np.array([[0, 0, 0], [1.5, 0, 0], [0, 2.25, 0], [0, 0, -3.125]], float)
TETRA_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)


def _meta(da):
    return dict(da.meta.items())


@pytest.mark.parametrize(
    "hemi,kind,primary,secondary,geom",
    [
        ("lh", "white", "CortexLeft", "GrayWhite", "Anatomical"),
        ("rh", "pial", "CortexRight", "Pial", "Anatomical"),
        ("lh", "inflated", "CortexLeft", None, "Inflated"),
        ("rh", "sphere", "CortexRight", None, "Sphere"),
    ],
)
def test_surf_gii_metadata(tmp_path, hemi, kind, primary, secondary, geom):
    write_surf_gii(tmp_path / "a.surf.gii", Mesh(TETRA_V, TETRA_F), hemi, kind)
    img = nib.load(tmp_path / "a.surf.gii")
    pts = img.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0]
    m = _meta(pts)
    assert m["AnatomicalStructurePrimary"] == primary
    assert m["GeometricType"] == geom
    assert m.get("AnatomicalStructureSecondary") == secondary
    r = read_gifti(tmp_path / "a.surf.gii")
    np.testing.assert_array_equal(r.vertices, TETRA_V)
    np.testing.assert_array_equal(r.faces, TETRA_F)


def test_surf_gii_rejects_bad_input(tmp_path):
    with pytest.raises(ValueError):
        write_surf_gii(tmp_path / "a.gii", Mesh(TETRA_V, TETRA_F), "lh", "bogus")
    with pytest.raises(ValueError):
        write_surf_gii(tmp_path / "a.gii", Mesh(TETRA_V, TETRA_F), "xx", "white")


def test_func_gii(tmp_path):
    data = np.arange(8, dtype=float).reshape(4, 2) / 3
    write_func_gii(tmp_path / "m.func.gii", data, "rh", names=["curv", "sulc"])
    img = nib.load(tmp_path / "m.func.gii")
    assert len(img.darrays) == 2
    assert _meta(img.darrays[1])["Name"] == "sulc"
    assert _meta(img.darrays[0])["AnatomicalStructurePrimary"] == "CortexRight"
    np.testing.assert_allclose(img.darrays[1].data, data[:, 1].astype(np.float32))
    write_func_gii(tmp_path / "v.func.gii", data[:, 0], "lh")
    assert len(nib.load(tmp_path / "v.func.gii").darrays) == 1


@pytest.mark.reference
@pytest.mark.parametrize("protocol", ["dkt31", "braincolor"])
def test_label_gii_matches_lut(tmp_path, protocol):
    d = OPT / "parc" / protocol
    table = read_protocol_table(d)
    assert not table.colors_from_fallback
    ids = np.array(table.ids())
    labels = np.resize(ids, 500)
    write_label_gii(tmp_path / "p.label.gii", labels, table, "lh")
    got, t2 = read_label_gii(tmp_path / "p.label.gii")
    np.testing.assert_array_equal(got, labels)
    # label table matches lut.csv exactly
    lut = np.loadtxt(d / "lut.csv", delimiter=",", skiprows=1, dtype=int)
    assert t2.ids() == sorted(set(lut[:, 0]) | set(table.ids()))
    for i, r, g, b in lut:
        assert t2.rgb(int(i)) == (r, g, b)
    # names come from label.csv
    assert t2.name(int(read_label_table(d / "label.csv").ids()[0])) != ""
    img = nib.load(tmp_path / "p.label.gii")
    assert _meta(img.darrays[0])["AnatomicalStructurePrimary"] == "CortexLeft"


def test_label_table_hsv_fallback(tmp_path):
    (tmp_path / "label.csv").write_text("label,ID\nfoo,3\nbar,7\nbaz,9\n")
    t = read_label_table(tmp_path / "label.csv")
    assert t.colors_from_fallback
    assert t.name(7) == "bar"
    assert len({t.rgb(i) for i in t.ids()}) == 3
    assert t.entries == read_label_table(tmp_path / "label.csv").entries  # deterministic
    assert hsv_fallback_colors([1, 2]) == hsv_fallback_colors([2, 1])


def test_label_gii_background_entry(tmp_path):
    (tmp_path / "label.csv").write_text("label,ID\nfoo,3\n")
    (tmp_path / "lut.csv").write_text("ID,R,G,B\n3,10,20,30\n")
    t = read_protocol_table(tmp_path)
    write_label_gii(tmp_path / "x.label.gii", np.array([0, 3, 3]), t, "rh")
    got, t2 = read_label_gii(tmp_path / "x.label.gii")
    assert t2.ids() == [0, 3] and t2.name(0) == "???" and t2.rgb(3) == (10, 20, 30)


def test_stats_tsv_sidecar(tmp_path):
    tsv, js = write_stats_tsv(
        tmp_path / "stats.tsv",
        {"roi": ["a", "b"], "area": np.array([1.5, np.nan]), "n": np.array([3, 4])},
        descriptions={"area": "surface area"},
        units={"area": "mm^2"},
    )
    cols = read_stats_tsv(tsv)
    assert cols == {"roi": ["a", "b"], "area": ["1.5", "n/a"], "n": ["3", "4"]}
    side = json.loads(js.read_text())
    assert side["area"] == {"Description": "surface area", "Units": "mm^2"}
    assert side["roi"]["Description"] == "roi"
    with pytest.raises(ValueError):
        write_stats_tsv(tmp_path / "bad.tsv", {"a": [1], "b": [1, 2]})
