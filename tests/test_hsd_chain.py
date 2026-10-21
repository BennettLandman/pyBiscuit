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
POLICY = {"E": 1e-3, "coeff": 1e-4, "ang_median": 0.01, "ang_p99": 0.1}  # docs/validation-tolerances.md (HSD rows)


def _bound(policy, spread, factor=1.5, floor=1e-6):
    """Acceptance bound: the policy value, or 1.5x the oracle's own spread under a one-ulp input change (stored in the
    golden) where that is larger.  A stable call is therefore held to the policy; a chaotic one to what the oracle itself shows."""
    return max(policy, factor * float(np.max(spread)) + floor)


def _coeff_of(path):
    t = Path(path).read_text().split()
    return np.array(t[1:3], np.float64), np.array(t[3:], np.float64).reshape(-1, 3).T


def _write_c(path, pole, coeff):
    from pybiscuit.sphere.spharm import ShDeformation, write_sh_coefficients

    write_sh_coefficients(str(path), ShDeformation(10, np.asarray(pole, np.float32), coeff))


@pytest.fixture(scope="module", params=list(CASES))
def chain_case(request, tmp_path_factory):
    name = request.param
    hemi, k, order = CASES[name]
    root = tmp_path_factory.mktemp(name)
    sd, td, jd = write_chain_case(root, hemi, warp_k=k)
    return name, order, ChainPaths(hemi, jd, td, sd), _g(name)


@pytest.mark.parametrize("call", [1, 2, 3, 4])
def test_each_call_vs_oracle(chain_case, call, capfd):
    """Call ``i`` of the port, started from the *oracle's* coefficient file after call ``i-1``, against the oracle's call ``i``."""
    name, order, paths, g = chain_case
    if call > 1:
        _write_c(paths.coeff, g[f"pole_{call - 1}"], g[f"coeff_{call - 1}"])
    elif paths.coeff.exists():
        paths.coeff.unlink()
    capfd.readouterr()
    run_chain(paths, rh_stage2=order, stages=[call])
    e = _energies(capfd.readouterr().out)
    pole, cf = _coeff_of(paths.coeff)
    gold_e = g[f"energies_{call}"]
    np.testing.assert_allclose(pole, g[f"pole_{call}"], atol=1e-6)
    assert cf.shape == (3, 121)  # degree 10 in every call
    e_rel = abs(e[-1] - gold_e[-1]) / gold_e[-1]
    ens_e = np.abs(g[f"ens_E_{call}"] - gold_e[-1]) / gold_e[-1]
    d_c = np.abs(cf - g[f"coeff_{call}"]).max()
    ens_c = np.abs(g[f"ens_coeff_{call}"] - g[f"coeff_{call}"]).max(axis=(1, 2))
    ens_it = np.abs(g[f"ens_iters_{call}"] - len(gold_e))
    msg = (name, call, f"E_rel {e_rel:.2e} (oracle spread {ens_e.max():.2e})", f"coeff {d_c:.2e} ({ens_c.max():.2e})", len(e), len(gold_e))
    assert e_rel <= _bound(POLICY["E"], ens_e), msg
    assert d_c <= _bound(POLICY["coeff"], ens_c, floor=2e-6), msg
    assert abs(len(e) - len(gold_e)) <= int(np.ceil(1.5 * ens_it.max())) + 1, msg  # diagnostic: same envelope rule
    if call == 4:
        ang = _angles(read_vtk(paths.registered).vertices, g["vertex"])
        assert np.median(ang) <= _bound(POLICY["ang_median"], g["ens_ang_med_4"], floor=1e-5), (name, np.median(ang))
        assert np.percentile(ang, 99) <= _bound(POLICY["ang_p99"], g["ens_ang_p99_4"], floor=1e-5), (name, np.percentile(ang, 99))


def test_chain_end_to_end_vs_oracle(chain_case, capfd):
    """The port's four calls chained on its own coefficient files against the oracle chain.  The oracle's own response to
    one float32 ulp in one call-1 input value (``e2e_*`` in the golden) bounds what counts as agreement."""
    name, order, paths, g = chain_case
    if paths.coeff.exists():
        paths.coeff.unlink()
    capfd.readouterr()
    run_chain(paths, rh_stage2=order)
    e = _energies(capfd.readouterr().out)
    ang = _angles(read_vtk(paths.registered).vertices, g["vertex"])
    msg = (name, f"median {np.median(ang):.3g} p99 {np.percentile(ang, 99):.3g}", "oracle", g["e2e_ang_med"], g["e2e_ang_p99"])
    assert np.median(ang) <= _bound(POLICY["ang_median"], g["e2e_ang_med"], floor=1e-5), msg
    assert np.percentile(ang, 99) <= _bound(POLICY["ang_p99"], g["e2e_ang_p99"], floor=1e-5), msg
    gold_e = g["energies_4"][-1]
    assert abs(e[-1] - gold_e) / gold_e <= _bound(POLICY["E"], np.abs(g["e2e_E"] - gold_e) / gold_e), msg


def test_golden_records_a_sensitive_chain():
    """The oracle itself is not reproducible to the policy bound: this is why the comparisons above are envelope-based."""
    spread = max(float(np.max(_g(n)["e2e_ang_med"])) for n in CASES)
    assert spread > POLICY["ang_median"]


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
    # recorded behaviour: the guess queries an AABB built on the raw radius-100 icomesh with unit vectors, so every
    # query misses, the guess's template mean map is 0 and the rotation it picks is arbitrary (but deterministic)
    from pybiscuit.hsd.guess import _stale_normals_tree
    from pybiscuit.hsd.icosahedron import icosahedron

    gv, _ = icosahedron(min(5, h.icosahedron))
    fid, _b = _stale_normals_tree(h.ico_v, h.ico_f, h.ico_v_raw).closest_face(gv)
    assert (fid < 0).all() and np.linalg.norm(h.ico_v, axis=1).min() > 99
    assert not np.any(g["coeff_all"][1, :, 1:]), "guess only sets degree-0 terms"
    assert np.abs(g["coeff_all"][1, :, 0]).max() > 1  # a large arbitrary rigid rotation, not the identity


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


def test_register_cli_runs_the_chain(tmp_path):
    from pybiscuit.cli import main

    sd, td, jd = write_chain_case(tmp_path, "lh")
    assert main(["register", "--surf-dir", str(jd), "--template-dir", str(td), "--sphere-dir", str(sd), "--hemi", "lh"]) == 0
    assert (jd / "lh.coeff.txt").read_text().split()[0] == "10"
    assert read_vtk(jd / "lh.res.sphere.reg.vtk").vertices.shape == (40962, 3)
