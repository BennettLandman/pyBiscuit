import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from conftest import MIPAV
from pybiscuit.config import load_config
from pybiscuit.io import read_asc
from pybiscuit.pipeline import InputError
from pybiscuit.pipeline.adapters import detect_adapter, find_inputs
from pybiscuit.pipeline.adapters.macruise_v3 import (
    apply_affine,
    find_macruise_v3,
    native_to_mni_matrix,
    read_mni_transform,
)
from pybiscuit.pipeline.adapters.pymacruise import REQUIRED as PYM_REQUIRED
from pybiscuit.pipeline.adapters.pymacruise import find_pymacruise
from pybiscuit.pipeline.model import sha256_file

PYM_RUN = MIPAV / "pymacruise/build/invivo-base-v3-dura002-no-regrid-2026-10-14"
LEGACY = MIPAV / "sample-macruise/output"
LEGACY_2 = MIPAV / "sample-macruise/100249"


# ------------------------------------------------------------------ synthetic runs
def _nii(path, shape=(6, 7, 8), affine=None, seed=0):
    aff = np.diag([1.0, 1.0, 1.0, 1.0]) if affine is None else affine
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(seed).integers(0, 5, shape).astype(np.int16)
    nib.save(nib.Nifti1Image(data, aff), str(path))


def _vtk(path, frame="RAS"):
    path.write_text(
        f"# vtk DataFile Version 3.0\nPyMaCRUISE coordinate_frame={frame} units=mm\nASCII\n"
        "DATASET POLYDATA\nPOINTS 3 float\n0 0 0\n1 0 0\n0 1 0\nPOLYGONS 1 4\n3 0 1 2\n"
    )


def _rec(p, **extra):
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size, **extra}


@pytest.fixture
def pym_run(tmp_path):
    root = tmp_path / "run"
    outs: dict[str, list] = {}
    layout = {
        "surface-thickness": ["inner_surface.vtk", "central_surface.vtk", "outer_surface.vtk"],
        "cruise-core": ["inner.nii.gz", "central.nii.gz", "outer.nii.gz"],
        "maseg-refinement": ["SEG_refine.nii.gz"],
    }
    for stage, files in layout.items():
        d = root / "stages" / stage / "attempt-01"
        d.mkdir(parents=True)
        outs[stage] = []
        for fn in files:
            p = d / fn
            _vtk(p) if fn.endswith(".vtk") else _nii(p)
            outs[stage].append(_rec(p))
    res = root / "resources"
    res.mkdir()
    _nii(res / "T1.nii.gz")
    run = {
        "schema_version": 1, "status": "complete", "converged": True,
        "spatial_convention": "RAS", "run_signature": "r" * 64,
        "implementation_signature": "i" * 64,
        "derived_resources": [_rec(res / "T1.nii.gz", role="t1")],
        "inputs": {}, "final_outputs": [],
        "stages": {s: {"attempt": 1, "status": "complete", "outputs": o}
                   for s, o in outs.items()},
    }
    (root / "run.json").write_text(json.dumps(run))
    return root


def _edit(root, fn):
    d = json.loads((root / "run.json").read_text())
    fn(d)
    (root / "run.json").write_text(json.dumps(d))


def test_pymacruise_synthetic_ok(pym_run):
    f = find_pymacruise(pym_run)
    assert f.ok and set(PYM_REQUIRED) <= set(f.products)
    assert f.source["run_signature"] == "r" * 64 and f.source["status"] == "complete"
    assert f.frames["inner_surface"] == "RAS"
    assert detect_adapter(pym_run) == "pymacruise"
    assert f.products["t1"].meta["shape"] == [6, 7, 8]


def test_pymacruise_moved_run_is_rebased(pym_run, tmp_path):
    moved = tmp_path / "moved"
    shutil.copytree(pym_run, moved)  # recorded paths still point into pym_run
    shutil.rmtree(pym_run / "stages")
    f = find_pymacruise(moved)
    assert f.ok and f.products["inner_surface"].path.startswith(str(moved.resolve()))


def test_pymacruise_incomplete_refused_unless_allowed(pym_run):
    _edit(pym_run, lambda d: d.update(status="running"))
    f = find_pymacruise(pym_run)
    assert not f.ok and "not 'complete'" in f.problems()[0] and "--allow-incomplete" in f.errors[0]
    g = find_pymacruise(pym_run, allow_incomplete=True)
    assert g.ok and any("(allowed)" in w for w in g.warnings)
    _edit(pym_run, lambda d: d["stages"]["cruise-core"].update(status="failed"))
    assert any("cruise-core" in e for e in find_pymacruise(pym_run).errors)


def test_pymacruise_hash_mismatch_and_missing(pym_run):
    p = pym_run / "stages/cruise-core/attempt-01/inner.nii.gz"
    _nii(p, seed=99)  # different content, same name
    f = find_pymacruise(pym_run)
    assert any("inner_levelset" in e and "SHA-256" in e for e in f.errors)
    p.unlink()
    assert any("inner_levelset" in e and "missing" in e for e in find_pymacruise(pym_run).errors)


def test_pymacruise_frame_and_grid_mismatch(pym_run):
    _vtk(pym_run / "stages/surface-thickness/attempt-01/outer_surface.vtk", frame="LPS")
    _edit(pym_run, lambda d: d["stages"]["surface-thickness"]["outputs"].__setitem__(
        2, _rec(pym_run / "stages/surface-thickness/attempt-01/outer_surface.vtk")))
    assert any("coordinate_frame=RAS" in e for e in find_pymacruise(pym_run).errors)


def test_pymacruise_grid_mismatch(pym_run):
    p = pym_run / "stages/maseg-refinement/attempt-01/SEG_refine.nii.gz"
    _nii(p, shape=(5, 5, 5))
    _edit(pym_run, lambda d: d["stages"]["maseg-refinement"]["outputs"].__setitem__(0, _rec(p)))
    assert any("grid" in e for e in find_pymacruise(pym_run).errors)


def test_pymacruise_non_ras_volume(pym_run):
    p = pym_run / "stages/cruise-core/attempt-01/central.nii.gz"
    _nii(p, affine=np.diag([-1.0, 1.0, 1.0, 1.0]))
    _edit(pym_run, lambda d: d["stages"]["cruise-core"]["outputs"].__setitem__(1, _rec(p)))
    errs = find_pymacruise(pym_run).errors
    assert any("central_levelset" in e and "expected RAS" in e for e in errs)


def test_pymacruise_missing_required_product(pym_run):
    _edit(pym_run, lambda d: d["stages"]["cruise-core"]["outputs"].pop(0))
    f = find_pymacruise(pym_run)
    assert "inner_levelset" in f.missing and not f.ok
    with pytest.raises(InputError, match="inner_levelset"):
        f.require_ok()


def test_not_a_run(tmp_path):
    with pytest.raises(InputError):
        find_pymacruise(tmp_path)
    with pytest.raises(InputError, match="cannot tell"):
        detect_adapter(tmp_path)


# ---------------------------------------------------------------- legacy synthetic
def _asc(path, shift=(0.0, 0.0, 0.0)):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float) + shift
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!ascii version of lh.white\n3 1\n"
        + "".join(f"{a} {b} {c} 0\n" for a, b, c in v)
        + "0 1 2 0\n"
    )


@pytest.fixture
def legacy_dir(tmp_path):
    out = tmp_path / "subj" / "output"
    for w in ("inner", "central", "outer"):
        _asc(out / f"Surfaces_FreeView/target_image_GMimg_{w}Surf.asc")
        _asc(out / f"Surfaces_MNI/target_image_GMimg_{w}Surf.asc", (2.0, 0, 0))
    ang = 0.1
    m = np.eye(4)
    m[:2, :2] = [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
    m[:3, 3] = [2, 0, 0]
    np.savetxt(out / "Surfaces_MNI/aff.txt", m, fmt="%.9f")
    _nii(out / "SegRefine/SEG_refine.nii.gz")
    _nii(out / "Raw/T1.nii.gz")
    return out


def test_legacy_synthetic_ok(legacy_dir):
    f = find_macruise_v3(legacy_dir.parent)  # parent dir with output/ is accepted
    assert f.ok and f.adapter == "macruise-v3" and Path(f.root).samefile(legacy_dir)
    assert f.frames["inner_surface"] == "RAS" and f.frames["inner_surface_mni"] == "MNI"
    assert f.products["inner_surface"].meta["n_vertices"] == 3
    assert detect_adapter(legacy_dir) == "macruise-v3"


def test_legacy_rejects_non_rigid_and_bad_counts(legacy_dir):
    np.savetxt(legacy_dir / "Surfaces_MNI/aff.txt", np.diag([2.0, 1, 1, 1]), fmt="%.6f")
    with pytest.raises(InputError, match="not a proper rotation"):
        read_mni_transform(legacy_dir / "Surfaces_MNI/aff.txt")
    assert any("rotation" in e for e in find_macruise_v3(legacy_dir).errors)
    (legacy_dir / "Surfaces_MNI/aff.txt").write_text("1 0 0\n")
    assert any("4x4" in e for e in find_macruise_v3(legacy_dir).errors)


def test_legacy_missing_mni(legacy_dir):
    (legacy_dir / "Surfaces_MNI/target_image_GMimg_innerSurf.asc").unlink()
    f = find_macruise_v3(legacy_dir)
    assert "inner_surface_mni" in f.missing and not f.ok


# --------------------------------------------------------------- FreeSurfer synthetic
pytestmark = pytest.mark.filterwarnings("ignore:.*volume information:UserWarning",
                                        "ignore:Unknown extension code:UserWarning")


@pytest.fixture
def fs_subject(tmp_path):
    root = tmp_path / "fs"
    (root / "surf").mkdir(parents=True)
    (root / "label").mkdir()
    (root / "mri").mkdir()
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    for h in ("lh", "rh"):
        for s in ("white", "pial", "inflated", "sphere", "sphere.reg"):
            nib.freesurfer.write_geometry(str(root / "surf" / f"{h}.{s}"), v + 1, faces)
        nib.freesurfer.write_morph_data(str(root / "surf" / f"{h}.curv"), np.zeros(4))
    return root


def test_freesurfer_synthetic(fs_subject):
    f = find_inputs(fs_subject, config=load_config())
    assert f.adapter == "freesurfer" and f.surf_recon == "freesurfer"
    assert f.ok and f.frames["lh.white"] == "tkRAS"
    assert f.products["rh.pial"].meta["n_faces"] == 4
    assert "lh.sphere.reg" in f.products and "lh.curv" in f.products
    assert any("orig.mgz" in w for w in f.warnings)
    (fs_subject / "surf" / "rh.sphere").unlink()
    assert "rh.sphere" in find_inputs(fs_subject).missing


def test_freesurfer_read_only(fs_subject):
    before = {p: p.stat().st_mtime_ns for p in fs_subject.rglob("*") if p.is_file()}
    find_inputs(fs_subject)
    assert before == {p: p.stat().st_mtime_ns for p in fs_subject.rglob("*") if p.is_file()}


# --------------------------------------------------------------- reference data
@pytest.mark.reference
def test_pymacruise_reference_run():
    f = find_pymacruise(PYM_RUN)
    assert f.ok, f.problems()
    expected = set(PYM_REQUIRED) | {"thickness_levelset", "inner_surface_thickness",
                                    "thickness_summary"}
    assert expected <= set(f.products)
    rec = json.loads((PYM_RUN / "run.json").read_text())
    assert f.source["run_signature"] == rec["run_signature"]
    assert f.source["status"] == "complete" and f.source["converged"] is True
    for name, st in {"inner_surface": "surface-thickness", "inner_levelset": "cruise-core",
                     "seg_refine": "maseg-refinement"}.items():
        assert f"stages/{st}/attempt-01" in f.products[name].path
    assert f.products["inner_levelset"].meta["shape"] == [184, 256, 256]
    assert f.products["t1"].meta["axis_codes"] == ["R", "A", "S"]


@pytest.mark.reference
@pytest.mark.parametrize("root", [LEGACY, LEGACY_2])
def test_legacy_reference_outputs(root):
    f = find_macruise_v3(root)
    assert f.ok, f.problems()
    for n in ("inner", "central", "outer"):
        assert f"{n}_surface" in f.products and f"{n}_surface_mni" in f.products
        assert f"{n}_levelset" in f.products
    assert f.products["inner_surface"].meta["n_vertices"] == 250874 or root == LEGACY_2
    assert f.frames["inner_surface"] == "RAS"


@pytest.mark.reference
def test_legacy_aff_maps_native_onto_mni():
    d = LEGACY
    native = read_asc(d / "Surfaces_FreeView/target_image_GMimg_innerSurf.asc", frame="RAS")
    mni = read_asc(d / "Surfaces_MNI/target_image_GMimg_innerSurf.asc", frame="RAS")
    assert np.array_equal(native.faces, mni.faces)
    m = read_mni_transform(d / "Surfaces_MNI/aff.txt")
    resid = np.abs(apply_affine(native_to_mni_matrix(m), native.vertices) - mni.vertices).max()
    assert resid <= 1e-4, resid
    # and aff.txt itself runs the other way
    assert np.abs(apply_affine(m, native.vertices) - mni.vertices).max() > 1.0
