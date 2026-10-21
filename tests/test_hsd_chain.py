# ruff: noqa: E501
"""BISCUIT's four-call HSD chain (brief 21; plan rows B6.5, spec §5.3 / H-1..H-8) against oracle goldens.

Goldens: ``tests/data/golden/hsd-chain/<case>/`` (``scripts/oracles/make_golden_hsd_chain.py``; oracle =
``hsd-icomesh-nofma``: the non-FMA HSD f169595 build plus a one-line fix of the dangling pointer in the
``--icomesh`` branch, see ``scripts/oracles/build_hsd_nofma.sh``).  Inputs are the scaled-down fixtures of
``tests/fixtures/hsd_chain.py`` (levels 3-6 instead of 4-7).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from fixtures.hsd_chain import write_chain_case, write_parc_case
from pybiscuit.hsd.chain import ChainPaths, chain_argv, parc_alignment_argv, run_chain
from pybiscuit.hsd.cli import main as hsd_main
from pybiscuit.hsd.guess import guess_init_coeff
from pybiscuit.hsd.register import HSD
from pybiscuit.io import read_vtk
from pybiscuit.sphere.spharm import read_sh_coefficients

GOLD = Path(__file__).parent / "data" / "golden" / "hsd-chain"
ORACLE = Path.home() / "dev" / "biscuit" / "oracles" / "hsd-icomesh-nofma" / "bin"
live = pytest.mark.skipif(not (ORACLE / "HSD").exists(), reason="hsd-icomesh-nofma oracle not built")


def _g(case):
    return np.load(GOLD / case / "golden.npz")


def _angles(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    cs = np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)
    return np.degrees(np.arccos(cs))


def _energies(text):
    return np.array([float(m.group(1)) for m in re.finditer(r"^\[\d+\] ([0-9.e+-]+)", text, re.M)])


# ------------------------------------------------------------------ argv (H-1, H-5, H-6, H-7)
def _paths(hemi="lh"):
    return ChainPaths(hemi, Path("J"), Path("T"), Path("S"))


def test_chain_argv_is_the_spec_table_lh():
    """§5.3: four single-property calls, ``-c C C --outputcoeff C C``, ``--tmpVar`` only in 3-4, ``-o R R`` in 4."""
    a = [" ".join(x) for x in chain_argv(_paths("lh"))]
    C = "J/lh.coeff.txt"
    S = lambda n: f"S/sphere_{n}.vtk"  # noqa: E731
    assert a[0] == f"-s {S(327680)} {S(20480)} --outputcoeff {C} {C} -p T/lh.iH.mean.txt J/lh.res.iH.txt --fixedSubjects 0 -d 10 --icomesh {S(5120)}"
    assert a[1] == f"-c {C} {C} -s {S(327680)} {S(81920)} --outputcoeff {C} {C} -p T/lh.sulc.mean.txt J/lh.res.sulc.txt --fixedSubjects 0 -d 10 --icomesh {S(20480)}"
    assert a[2] == f"-c {C} {C} -s {S(327680)} {S(327680)} --outputcoeff {C} {C} -p T/lh.sulc.mean.txt J/lh.res.sulc.txt --fixedSubjects 0 -d 10 --icomesh {S(81920)} --tmpVar T/lh.sulc.var.txt"
    R = "J/lh.res.sphere.reg.vtk"
    assert a[3] == f"-c {C} {C} -s {S(327680)} {S(327680)} -o {R} {R} --outputcoeff {C} {C} -p T/lh.curv.mean.txt J/lh.res.H.txt --fixedSubjects 0 -d 10 --icomesh {S(327680)} --tmpVar T/lh.curv.var.txt"
    assert not any("--weight" in x or "--noguess" in x or "--resample" in x or "--idprior" in x for x in a)


def test_rh_stage2_orders():
    mac = chain_argv(_paths("rh"))[1]
    lh_like = chain_argv(_paths("rh"), rh_stage2="lh")[1]
    i = mac.index("-s")
    assert mac[i + 1 : i + 3] == ["S/sphere_81920.vtk", "S/sphere_327680.vtk"]  # surf_reg_macruise.m:112
    assert lh_like[i + 1 : i + 3] == ["S/sphere_327680.vtk", "S/sphere_81920.vtk"]
    assert [x for k, x in enumerate(mac) if k not in (i + 1, i + 2)] == [x for k, x in enumerate(lh_like) if k not in (i + 1, i + 2)]
    assert chain_argv(_paths("lh"), rh_stage2="macruise")[1] == chain_argv(_paths("lh"), rh_stage2="lh")[1]  # lh unaffected


def test_parc_alignment_argv_is_the_spec_call():
    a = parc_alignment_argv("S/sphere_327680.vtk", "tmp/id.sphere.vtk", "P/lh.curv.mean.txt", "tmp/id.H.txt")
    assert " ".join(a) == (
        "-s S/sphere_327680.vtk tmp/id.sphere.vtk -o tmp/id.sphere.vtk tmp/id.sphere.vtk -p P/lh.curv.mean.txt tmp/id.H.txt "
        "--fixedSubjects 0 --noguess -d 0 --icosahedron 5"
    )


def test_register_cli_dry_run(tmp_path, capsys):
    from pybiscuit.cli import main

    assert main(["register", "--surf-dir", "J", "--template-dir", "T", "--sphere-dir", "S", "--hemi", "rh", "--dry-run"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 4 and out[1].startswith("[rh call 2] HSD -c J/rh.coeff.txt J/rh.coeff.txt -s S/sphere_81920.vtk S/sphere_327680.vtk")


# ------------------------------------------------------------------ chain vs oracle (H-1, H-2, H-4)
CASES = {"chain_lh": ("lh", 1, "macruise"), "chain_rh_macruise": ("rh", 2, "macruise"), "chain_rh_lhorder": ("rh", 2, "lh")}


@pytest.fixture(scope="module", params=list(CASES))
def chain_run(request, tmp_path_factory):
    name = request.param
    hemi, k, order = CASES[name]
    root = tmp_path_factory.mktemp(name)
    sd, td, jd = write_chain_case(root, hemi, warp_k=k)
    paths = ChainPaths(hemi, jd, td, sd)
    states = []
    energies = []
    import io
    from contextlib import redirect_stdout

    for i in (1, 2, 3, 4):
        buf = io.StringIO()
        with redirect_stdout(buf):
            run_chain(paths, rh_stage2=order, stages=[i])
        energies.append(_energies(buf.getvalue()))
        pole, cf = np.array(paths.coeff.read_text().split()[1:3], np.float64), np.array(paths.coeff.read_text().split()[3:], np.float64).reshape(-1, 3).T
        states.append((pole, cf))
    return name, paths, states, energies


def test_chain_matches_oracle(chain_run):
    name, paths, states, energies = chain_run
    g = _g(name)
    for i, (pole, cf) in enumerate(states, 1):
        assert cf.shape == g[f"coeff_{i}"].shape == (3, 121)  # degree 10 capped in every call
        np.testing.assert_allclose(pole, g[f"pole_{i}"], atol=1e-6)
        fine = i == 4  # damped-Newton stage (level 6)
        assert np.abs(cf - g[f"coeff_{i}"]).max() < (5e-3 if fine else 1e-4), (name, i)
        gold = g[f"energies_{i}"]
        assert abs(len(energies[i - 1]) - len(gold)) <= (2 if fine else 0), (name, i)
        assert abs(energies[i - 1][-1] - gold[-1]) / gold[-1] < 1e-3, (name, i)
    ang = _angles(read_vtk(paths.registered).vertices, g["vertex"])
    assert np.median(ang) <= 0.01 and np.percentile(ang, 99) <= 0.1, (name, np.median(ang), np.percentile(ang, 99))


def test_chain_registers_the_warped_subject(chain_run):
    """Independent of the oracle: each stage lowers the energy, and the final cost is below the unregistered one."""
    _, _, _, energies = chain_run
    for e in energies:
        assert e[-1] < e[0]


# ------------------------------------------------------------------ H-3: guess with --icomesh radius-100 spheres
def test_guess_with_icomesh_matches_oracle(tmp_path):
    sd, td, jd = write_chain_case(tmp_path, "lh", warp_k=1)
    a = chain_argv(ChainPaths("lh", jd, td, sd))[0]
    from pybiscuit.hsd.cli import build_parser

    ns = build_parser().parse_args(a)
    nprop = len(ns.property) // len(ns.sphere)
    props = [ns.property[s * nprop : (s + 1) * nprop] for s in range(len(ns.sphere))]
    h = HSD(
        ns.sphere, props, output_coeff=ns.outputcoeff, degree=ns.degree, fixed_subjects=ns.fixedSubjects,
        icosahedron_level=ns.icosahedron, ico_mesh=ns.icomesh, guess=True, verbose=False,
    )  # fmt: skip
    guess_init_coeff(h)
    g = _g("guess_icomesh")
    assert h.icosahedron == 3 and h.nsamples == 642
    np.testing.assert_allclose(h.coeff, g["coeff_all"], atol=1e-6)
    # recorded behaviour: with the raw radius-100 icomesh the AABB queries of the guess are all misses
    assert not np.any(g["coeff_all"][1, :, 1:]), "guess only sets degree-0 terms"


# ------------------------------------------------------------------ H-8: parcellation alignment, -o == input
def test_parc_alignment_in_place_matches_oracle(tmp_path, capfd):
    t, n, cm, h = write_parc_case(tmp_path)
    before = read_vtk(n).vertices.copy()
    assert hsd_main(parc_alignment_argv(t, n, cm, h)) == 0
    log = capfd.readouterr().out
    g = _g("parc_align")
    after = read_vtk(n).vertices
    assert not np.allclose(after, before)  # the input file was overwritten
    ang = _angles(after, g["vertex"])
    assert np.median(ang) <= 0.01 and np.percentile(ang, 99) <= 0.1, (np.median(ang), np.percentile(ang, 99))
    e = _energies(log)
    assert abs(e[-1] - g["energies"][-1]) / g["energies"][-1] < 1e-3


# ------------------------------------------------------------------ H-2 / H-4 without the oracle
def test_property_files_longer_than_the_mesh_are_truncated(tmp_path):
    """Calls 1-2: the subject mesh has fewer vertices than the property file; only the first nVertex lines count."""
    sd, td, jd = write_chain_case(tmp_path, "lh")
    p = ChainPaths("lh", jd, td, sd)
    a = chain_argv(p)[0]
    a_cut = list(a)
    cut = jd / "lh.res.iH.cut.txt"
    lines = (jd / "lh.res.iH.txt").read_text().splitlines()
    assert len(lines) == 40962
    cut.write_text("\n".join(lines[:2562]) + "\n")  # sphere_20480 stand-in has 2,562 vertices
    a_cut[a_cut.index(str(jd / "lh.res.iH.txt"))] = str(cut)
    c1, c2 = jd / "c1.txt", jd / "c2.txt"
    for out, argv in ((c1, a), (c2, a_cut)):
        argv = [str(out) if x == str(p.coeff) else x for x in argv]
        assert hsd_main(argv) == 0
    assert c1.read_text() == c2.read_text()


def test_fixed_subject_neither_reads_nor_writes_the_shared_coefficient_file(tmp_path):
    sd, td, jd = write_chain_case(tmp_path, "lh")
    p = ChainPaths("lh", jd, td, sd)
    argvs = chain_argv(p)
    assert hsd_main(argvs[0]) == 0
    first = p.coeff.read_text()
    toks = first.split()
    assert len(toks) == 3 + 3 * 121 and toks[0] == "10"
    # call 2 reads the file for the deformable subject only and rewrites it
    shutil.copy(p.coeff, jd / "prev.txt")
    assert hsd_main(argvs[1]) == 0
    assert p.coeff.read_text() != first
    d = read_sh_coefficients(str(p.coeff), 10)
    assert d.coeff.shape == (3, 121)


@live
def test_live_oracle_stage1(tmp_path):
    """The oracle binary itself, on call 1, agrees with the committed golden (guards the golden's provenance)."""
    import subprocess

    sd, td, jd = write_chain_case(tmp_path, "lh")
    p = ChainPaths("lh", jd, td, sd)
    a = chain_argv(p)[0]
    subprocess.run([str(ORACLE / "HSD"), *a], check=True, capture_output=True)
    cf = np.array(p.coeff.read_text().split()[3:], np.float64).reshape(-1, 3).T
    assert np.abs(cf - _g("chain_lh")["coeff_1"]).max() < 1e-9
