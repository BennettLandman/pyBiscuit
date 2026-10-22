"""FreeSurfer-route HSD chain (``surf_reg_fs2.m``, spec §12): argv vs the MaCRUISE chain."""

from pathlib import Path

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
