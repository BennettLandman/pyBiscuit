import json

import numpy as np
import pytest

from pybiscuit.cli import main
from pybiscuit.io import (
    read_gifti,
    read_surface,
    surface_format,
    write_fs_surf,
    write_func_gii,
    write_gifti,
    write_surf_gii,
    write_surface,
    write_vtk,
)
from pybiscuit.mesh.info import describe, surface_info
from pybiscuit.mesh.model import Mesh
from pybiscuit.sphere.icosphere import icosphere


def _tetra(**kw):
    v = np.array([[0, 0, 0], [1.5, 0, 0], [0, 2.25, 0], [0, 0, -3.125]], float)  # float32-exact
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)  # outward
    return Mesh(v, f, **kw)


@pytest.mark.parametrize(
    ("name", "fmt"),
    [("m.vtk", "vtk"), ("m.surf.gii", "gifti"), ("m.asc", "asc"), ("lh.white", "freesurfer")],
)
def test_read_surface_dispatch(tmp_path, name, fmt):
    m = _tetra(frame="RAS")
    p = tmp_path / name
    write_surface(p, m, format="freesurfer" if fmt == "freesurfer" else None)
    assert surface_format(p) == fmt
    r = read_surface(p)
    np.testing.assert_allclose(r.vertices, m.vertices, atol=1e-5)
    np.testing.assert_array_equal(r.faces, m.faces)
    assert r.meta["format"] == fmt
    assert r.frame == {"freesurfer": "tkRAS"}.get(fmt, "RAS")


def test_surface_format_rejects_unknown(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("1 2 3\n")
    with pytest.raises(ValueError, match="unrecognised"):
        surface_format(p)
    with pytest.raises(ValueError, match="cannot infer"):
        write_surface(p, _tetra())


def test_gifti_frame_and_metadata(tmp_path):
    write_surf_gii(tmp_path / "lh.pial.surf.gii", _tetra(), hemi="lh", kind="pial")
    r = read_gifti(tmp_path / "lh.pial.surf.gii")
    assert r.frame == "RAS"
    g = r.meta["gifti"]
    assert g["AnatomicalStructurePrimary"] == "CortexLeft"
    assert g["AnatomicalStructureSecondary"] == "Pial"
    assert g["GeometricType"] == "Anatomical"
    assert g["dataspace"] == "NIFTI_XFORM_SCANNER_ANAT"
    # write_gifti keeps a known frame and leaves an unknown one unset
    write_gifti(tmp_path / "a.gii", _tetra(frame="MNI"))
    assert read_gifti(tmp_path / "a.gii").frame == "MNI"
    write_gifti(tmp_path / "b.gii", _tetra())
    assert read_gifti(tmp_path / "b.gii").frame is None


def test_surface_info_closed_sphere():
    m = icosphere(3)
    info = surface_info(m)
    t, g = info["topology"], info["geometry"]
    assert info["counts"] == {"vertices": 642, "faces": 1280, "edges": 1920}
    assert t["euler_characteristic"] == 2 and t["genus"] == 0 and t["components"] == 1
    assert t["closed"] and t["manifold"] and t["consistently_oriented"]
    assert g["normals"] == "outward"
    assert g["area"] == pytest.approx(4 * np.pi, rel=0.02)
    assert g["volume"] == pytest.approx(4 / 3 * np.pi, rel=0.03)


def test_surface_info_open_with_data():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], float)
    m = Mesh(v, [[0, 1, 2]], {"thick": np.array([1.0, np.nan, 3.0, 2.0])})
    info = surface_info(m)
    t = info["topology"]
    assert not t["closed"] and t["boundary_loops"] == 1 and t["unreferenced_vertices"] == 1
    assert "volume" not in info["geometry"]
    d = info["point_data"]["thick"]
    assert (d["min"], d["max"], d["non_finite"]) == (1.0, 3.0, 1)


def test_describe_data_only_gifti(tmp_path):
    p = tmp_path / "lh.curv.func.gii"
    write_func_gii(p, np.arange(4.0), hemi="lh", names=["curv"])
    info = describe(p)
    assert info["kind"] == "data"
    assert info["arrays"][0]["name"] == "curv" and info["arrays"][0]["max"] == 3.0


def test_surf_info_cli(tmp_path, capsys):
    write_surf_gii(tmp_path / "lh.white.surf.gii", icosphere(2), hemi="lh", kind="white")
    write_vtk(_tetra(), tmp_path / "t.vtk")
    write_fs_surf(tmp_path / "rh.white", _tetra())
    files = [str(tmp_path / n) for n in ("lh.white.surf.gii", "t.vtk", "rh.white")]
    assert main(["surf-info", *files]) == 0
    out = capsys.readouterr().out
    assert "CortexLeft" in out and "genus: 0" in out and "format: freesurfer" in out
    assert main(["surf-info", "--json", files[0]]) == 0
    (info,) = json.loads(capsys.readouterr().out)
    assert info["counts"]["vertices"] == 162 and info["frame"] == "RAS"
    assert main(["surf-info", str(tmp_path / "missing.vtk")]) == 1
