# ruff: noqa: E501
"""FreeSurfer-route HSD chain (``surf_reg_fs2.m``, spec §12): argv vs the MaCRUISE chain."""

from pathlib import Path

import numpy as np
import pytest

from pybiscuit.hsd.chain import ChainPaths, chain_argv


def _paths(hemi):
    return ChainPaths(hemi, Path("J"), Path("T"), Path("S"))


def _prop(argv):
    i = argv.index("-p")
    return argv[i + 1 : i + 3]


@pytest.mark.parametrize("hemi", ["lh", "rh"])
def test_fs_route_features(hemi):
    fs, mac = chain_argv(_paths(hemi), route="fs"), chain_argv(_paths(hemi))
    assert [_prop(a) for a in fs] == [
        [f"T/{hemi}.iH.mean.txt", f"J/{hemi}.res.H.txt"],
        [f"T/{hemi}.sulc.mean.txt", f"J/{hemi}.res.sulc.txt"],
        [f"T/{hemi}.sulc.mean.txt", f"J/{hemi}.res.sulc.txt"],
        [f"T/{hemi}.curv.mean.txt", f"J/{hemi}.res.curv.txt"],
    ]
    for i in (1, 2):  # calls 2-3 identical
        assert fs[i] == mac[i] or hemi == "rh"


def test_fs_route_rh_stage2_has_lh_order():
    fs = chain_argv(_paths("rh"), route="fs")[1]
    i = fs.index("-s")
    assert fs[i + 1 : i + 3] == ["S/sphere_327680.vtk", "S/sphere_81920.vtk"]
    same = chain_argv(_paths("rh"), route="fs", rh_stage2="macruise")
    assert chain_argv(_paths("rh"), route="fs") == same


def test_route_validated():
    with pytest.raises(ValueError):
        chain_argv(_paths("lh"), route="civet")


# --------------------------------------------------- guess with --icomesh (image evidence, brief 22 / Q17)
def _guess_coeff(tmp_path, *, unit: bool):
    from fixtures.hsd_chain import write_chain_case
    from pybiscuit.hsd.cli import build_parser
    from pybiscuit.hsd.guess import guess_init_coeff
    from pybiscuit.hsd.register import HSD

    sd, td, jd = write_chain_case(tmp_path, "lh", warp_k=1)
    ns = build_parser().parse_args(chain_argv(ChainPaths("lh", jd, td, sd))[0])
    nprop = len(ns.property) // len(ns.sphere)
    props = [ns.property[s * nprop : (s + 1) * nprop] for s in range(len(ns.sphere))]
    h = HSD(
        ns.sphere, props, output_coeff=ns.outputcoeff, degree=ns.degree,
        fixed_subjects=ns.fixedSubjects, icosahedron_level=ns.icosahedron, ico_mesh=ns.icomesh,
        guess=True, verbose=False, guess_unit_ico=unit,
    )  # fmt: skip
    guess_init_coeff(h)
    return h.coeff[1, :, 0].copy()


def test_guess_with_icomesh_finds_no_rotation_for_an_aligned_subject(tmp_path):
    """The image's first HSD call registers an aligned FreeSurfer sphere without a big spurious rotation
    (open-examples: displacement median 10 deg).  Upstream's raw radius-100 mesh makes every mean-map query
    miss, and its guess then picks a 16 rad-scale arbitrary rotation; the default queries the unit mesh."""
    unit = _guess_coeff(tmp_path / "u", unit=True)
    raw = _guess_coeff(tmp_path / "r", unit=False)
    assert np.abs(raw).max() > 1  # upstream: arbitrary rigid rotation (golden guess_icomesh)
    assert np.abs(unit).max() < 0.5, unit
