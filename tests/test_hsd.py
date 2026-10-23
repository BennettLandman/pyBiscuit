# ruff: noqa: E501
"""HSD port (plan rows B6.1, B6.3-B6.6) against goldens from the non-FMA upstream oracle.

Goldens: ``tests/data/golden/hsd/<case>/`` (``scripts/oracles/make_golden_hsd.py``; oracle = HSD
f169595 + MeshLib dcef807, Apple clang ``-ffp-contract=off``, Accelerate BLAS/LAPACK).  The live-oracle
tests are skipped when ``~/dev/biscuit/oracles/hsd-nofma`` is absent.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fixtures.hsd_cases import EVAL_CASES, RUN_CASES, coeff_states, write_case
from pybiscuit.hsd.cli import build_parser
from pybiscuit.hsd.cli import main as hsd_main
from pybiscuit.hsd.icosahedron import icosahedron, icosahedron_counts
from pybiscuit.hsd.register import HSD, _patch_cache
from pybiscuit.io import read_vtk

GOLD = Path(__file__).parent / "data" / "golden" / "hsd"
ORACLE = Path.home() / "dev" / "biscuit" / "oracles" / "hsd-nofma" / "bin"
live = pytest.mark.skipif(not (ORACLE / "HSD").exists(), reason="hsd-nofma oracle not built")


def _g(case):
    return np.load(GOLD / case / "golden.npz")


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ----------------------------------------------------------------------------- B6.3
@pytest.mark.parametrize("level", [3, 4, 5])
def test_icosahedron_matches_oracle_bitwise(level):
    g = _g("init")
    v, f = icosahedron(level)
    assert np.array_equal(v, g[f"samples_{level}"])
    assert np.array_equal(f, g[f"faces_{level}"])


@pytest.mark.parametrize("level", [0, 1, 2, 3, 6])
def test_icosahedron_structure(level):
    v, f = icosahedron(level)
    nv, nf = icosahedron_counts(level)
    assert v.shape == (nv, 3) and f.shape == (nf, 3)
    assert np.abs(np.linalg.norm(v, axis=1) - 1).max() < 1e-6
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    assert (cnt == 2).all()  # closed manifold
    vol = np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum()
    assert vol > 0  # outward orientation


def test_patch_cache_adjacency():
    _, f = icosahedron(2)
    pc = _patch_cache(f)
    assert (pc >= 0).all()
    for i in range(len(f)):
        for k in range(3):
            j = pc[i, k]
            a, b = f[i, k], f[i, (k + 1) % 3]
            assert any((f[j, m], f[j, (m + 1) % 3]) == (b, a) for m in range(3))


@pytest.fixture(scope="module")
def case_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("hsd")
    write_case(d, 3)
    return d


def _subjects(d, n):
    return [str(d / f"s{k}.sphere.vtk") for k in range(n)], [[str(d / f"s{k}.curv.txt")] for k in range(n)]


def test_init_state_matches_oracle(case_dir):
    """Bit-identical on the machine that made the goldens; elsewhere (libm ``atan2f`` ulp) a sample on
    a face edge/vertex may pick the neighbouring face: tolerated for <= 1 % of the samples if the
    interpolated feature agrees (docs/validation-tolerances.md, tie policy)."""
    g = _g("init")
    sph, prop = _subjects(case_dir, 3)
    h = HSD(sph, prop, degree=6, icosahedron_level=4, verbose=False)
    ties = 0
    for s in range(3):
        sub = h.subj[s]
        assert np.array_equal(sub.v, g[f"vertex_{s}"])
        assert np.array_equal(sub.prop, g[f"property_{s}"])
        assert np.array_equal(sub.area0, g[f"area0_{s}"])
        ties += int((sub.tree_cache != g[f"tree_cache_{s}"]).sum())
    assert ties <= 0.01 * 3 * h.nsamples, ties
    assert np.abs(h.feature - g["feature"]).max() < 1e-5
    assert np.abs(h.mean - g["mean"]).max() < 1e-5
    assert np.array_equal(h.variance, g["variance"])


@pytest.mark.bitexact
def test_init_state_bitexact(case_dir):
    g = _g("init")
    sph, prop = _subjects(case_dir, 3)
    h = HSD(sph, prop, degree=6, icosahedron_level=4, verbose=False)
    for s in range(3):
        assert np.array_equal(h.subj[s].tree_cache, g[f"tree_cache_{s}"])
    assert np.array_equal(h.feature, g["feature"])
    assert np.array_equal(h.mean, g["mean"])


# ------------------------------------------------------------------ B6.4 (unit level)
def _eval_python(c, d):
    sph, prop = _subjects(d, 3)
    h = HSD(sph, prop, degree=c["deg"], icosahedron_level=c["ico"], guess=False, verbose=False)
    A, B = coeff_states(c["deg"], c["deginc"], 3)
    for sub in h.subj:
        sub.step = 1.0
    h.degree_inc, h.n_success, h.n_iter = c["deginc"], 2, 0
    h.coeff[:] = A
    if c["trans"] == 1:
        for s in range(3):
            h.update_deformation(s, True)
        h.update_properties()
        for sub in h.subj:
            h._init_flips(sub)
        h.update_displacement(h.degree)
        h.update_property_stats()
    elif c["trans"] == 2:
        h.update_property_stats()
    EA = h.cost()
    eA = [EA, h.cost_prop, h.cost_area, h.cost_disp]
    vA = [s.v.copy() for s in h.subj]
    feat = (h.feature.copy(), h.mean.copy(), h.variance.copy())
    h.update_gradient(c["beg"], c["end"], 0.001)
    # raw gradient and Hessian are recomputed without the solve for comparison
    gdir = h.gradient.copy()
    h.coeff[:] = B
    EB = h.cost()
    return dict(eA=eA, eB=[EB, h.cost_prop, h.cost_area, h.cost_disp], gdir=gdir, vA=vA, feat=feat, h=h)


@pytest.mark.parametrize("name", list(EVAL_CASES))
def test_unit_level_vs_oracle(name, case_dir):
    c, g = EVAL_CASES[name], _g(name)
    r = _eval_python(c, case_dir)
    for key, gold in (("eA", g["energy_A"]), ("eB", g["energy_B"])):
        assert _rel(r[key], gold[:4]) < 2e-7, (name, key, r[key], gold[:4])
    assert not gold[4:].any()  # no flips in these states
    for s in range(3):
        scale = np.abs(g[f"gradient_dir_{s}"]).max()
        assert np.abs(r["gdir"][s] - g[f"gradient_dir_{s}"]).max() < 2e-5 * scale, (name, s)
        assert np.abs(r["vA"][s] - g[f"vertex_A_{s}"]).max() < 3e-7  # float32 ulp noise from libm
    if "feature_A" in g:
        assert np.abs(r["feat"][0] - g["feature_A"]).max() < 1e-6
        assert np.abs(r["feat"][1] - g["mean_A"]).max() < 1e-6


@pytest.mark.parametrize("name", ["newton_band3", "newton_final"])
def test_hessian_and_raw_gradient_vs_oracle(name, case_dir):
    """J^T D J and the raw (undamped) gradient before ``dsysv`` (fine level)."""
    from pybiscuit.hsd import gradient as Gm

    c, g = EVAL_CASES[name], _g(name)
    r = _eval_python(c, case_dir)
    h = r["h"]
    h.coeff[:] = coeff_states(c["deg"], c["deginc"], 3)[0]
    h.cost()
    # recompute pieces for subject 0 exactly as update_gradient does
    sub, s = h.subj[0], 0
    n, n0 = (c["end"] + 1) ** 2, c["beg"] ** 2
    size = n - n0
    norm_p = 1.0 / float((h.nprop * h.nsamples) * h.nsubj)
    gp, Hp = Gm.accumulate_properties(
        sub, 0, c["beg"], c["end"], samples=h.samples, mean_k=h.mean[0], var_k=h.variance[0],
        feat_k=h.feature[s, 0], normalization=norm_p, fine=True, unit_var=False, axis0=sub.axis0,
        ring=h._ring(sub),
    )  # fmt: skip
    H = Hp
    raw = np.zeros((3, h.K))
    for comp in range(3):
        raw[comp, n0:n] = gp[comp * size : (comp + 1) * size] * norm_p
    if c["deginc"] > 0:
        gd, Hd = Gm.accumulate_displacement(
            sub, c["beg"], c["end"], normalization=h.lambda2 / 3.0, fine=True, axis0=sub.axis0
        )
        H = H + Hd
        for comp in range(3):
            raw[comp, n0:n] += gd[comp * size : (comp + 1) * size] * (h.lambda2 / 3.0) / len(sub.v)
    gold = g["hessian_0"]
    assert _rel(H, gold) < 1e-5 or _rel(H.T, gold) < 1e-5
    assert _rel(raw, g["gradient_raw_0"]) < 1e-5


# ------------------------------------------------------------------ full runs (B6.5)
def _run_port(name, c, d):
    nsub = c["nsub"]
    out = d / f"port_{name}"
    out.mkdir(exist_ok=True)
    a = ["-s"] + [str(d / f"s{k}.sphere.vtk") for k in range(nsub)]
    a += ["-p"] + [str(d / f"s{k}.curv.txt") for k in range(nsub)]
    a += ["-o"] + [str(out / f"{k}.reg.vtk") for k in range(nsub)]
    a += ["--outputcoeff"] + [str(out / f"{k}.coeff") for k in range(nsub)]
    flags = [str(d / "tmpvar.txt") if f == "tmpvar.txt" else f for f in c["flags"]]
    return out, a + flags


def _angles(a, b):
    cs = np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)
    return np.degrees(np.arccos(cs))


@pytest.mark.parametrize("name", list(RUN_CASES))
def test_full_run_vs_oracle(name, tmp_path, capfd):
    c, g = RUN_CASES[name], None
    g = _g(name)
    write_case(tmp_path, c["nsub"])
    out, argv = _run_port(name, c, tmp_path)
    assert hsd_main(argv) == 0
    log = capfd.readouterr().out
    en = np.array([float(m.group(1)) for m in re.finditer(r"^\[\d+\] ([0-9.e+-]+)", log, re.M)])
    gold = g["energies"]
    fine = "6" in c["flags"][c["flags"].index("--icosahedron") + 1]
    if fine:  # damped Newton path: an iteration more or less is allowed (acceptance: 0.1 % energy)
        assert abs(len(en) - len(gold)) <= 2
    else:
        assert len(en) == len(gold)
    assert abs(en[-1] - gold[-1]) / gold[-1] < 1e-3
    for k in range(c["nsub"]):
        if f"vertex_{k}" not in g:
            assert not (out / f"{k}.reg.vtk").exists()  # fixed subjects are not written
            continue
        ang = _angles(read_vtk(out / f"{k}.reg.vtk").vertices, g[f"vertex_{k}"].astype(np.float64))
        assert np.median(ang) <= 0.01 and np.percentile(ang, 99) <= 0.1, (name, k)
        cf = np.array((out / f"{k}.coeff").read_text().split()[3:], dtype=np.float64).reshape(-1, 3).T
        assert np.abs(cf - g[f"coeff_{k}"]).max() < (5e-3 if fine else 1e-5)


def test_registration_reduces_feature_variance(tmp_path):
    """Sanity independent of the oracle: the warped subjects agree better after registration."""
    write_case(tmp_path, 3)
    sph, prop = _subjects(tmp_path, 3)
    h = HSD(sph, prop, degree=6, icosahedron_level=5, verbose=False)
    h.optimization()
    assert h.history[-1][1] < 0.95 * h.history[0][1]  # 0.0125 -> 0.0111 with degree 6


# ------------------------------------------------------------------------- CLI
def test_cli_flags_match_parse_args_h():
    src = Path("~/dev/biscuit/reference/repos/HSD/wrapper/PARSE_ARGS.h").expanduser()
    if not src.exists():
        pytest.skip("reference/repos/HSD not available")
    upstream = set()
    text = re.sub(r"/\*.*?\*/", "", src.read_text(), flags=re.S)  # commented-out flags
    for line in text.splitlines():
        if line.strip().startswith("//") or "nStreams" in line:  # nStreams: CUDA build only
            continue
        m = re.search(r'add_(?:option|flag)\("([^"]+)"', line)
        if m:
            upstream.update(m.group(1).split(","))
    ours = {s for a in build_parser()._actions for s in a.option_strings if s != "-h" and s != "--help"}
    ours -= {"--guess-raw-icomesh", "--robust-aabb"}  # port-only switches
    assert upstream == ours


def test_cli_errors(tmp_path, capsys):
    write_case(tmp_path, 3)
    s = [str(tmp_path / f"s{k}.sphere.vtk") for k in range(3)]
    p = [str(tmp_path / f"s{k}.curv.txt") for k in range(3)]
    base = ["-s", *s, "-p", *p]
    assert hsd_main(base + ["-o", "a", "b"]) == 1  # outputs inconsistent
    assert "incosistent with # of outputs" in capsys.readouterr().out
    assert hsd_main(base + ["-o", "a", "b", "c", "--weight", "1", "1"]) == 1
    assert hsd_main(base + ["-o", "a", "b", "c", "--fixedSubjects", "0", "1", "2"]) == 1
    assert hsd_main(base + ["-o", "a", "b", "c", "--fixedSubjects", "7"]) == 1


# ------------------------------------------------------------------ live oracle
@live
def test_live_oracle_icosahedron_levels():
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write_case(d, 2)
        for lvl in (3, 6):
            (d / f"o{lvl}").mkdir()
            env = {"HSD_H_MODE": "init", "HSD_H_OUT": str(d / f"o{lvl}")}
            subprocess.run(
                [str(ORACLE / "hsd_harness"), "-s", "s0.sphere.vtk", "s1.sphere.vtk", "-p", "s0.curv.txt",
                 "s1.curv.txt", "-d", "2", "--icosahedron", str(lvl)],
                cwd=d, env=env, check=True, capture_output=True,
            )  # fmt: skip
            v, f = icosahedron(lvl)
            assert np.array_equal(v, np.load(d / f"o{lvl}" / "samples.npy"))
            assert np.array_equal(f, np.load(d / f"o{lvl}" / "ico_faces.npy"))


def test_writecoeff_and_coefficient_restart(tmp_path):
    """``--writecoeff`` writes coefficients during the run; ``-c`` reads them back (degree cropped)."""
    write_case(tmp_path, 2)
    s = [str(tmp_path / f"s{k}.sphere.vtk") for k in range(2)]
    p = [str(tmp_path / f"s{k}.curv.txt") for k in range(2)]
    cf = [str(tmp_path / f"c{k}.txt") for k in range(2)]
    base = ["-s", *s, "-p", *p, "--icosahedron", "4", "--noguess", "--outputcoeff", *cf]
    assert hsd_main(base + ["-d", "3", "--writecoeff"]) == 0
    toks = Path(cf[0]).read_text().split()
    assert toks[0] == "3" and len(toks) == 3 + 3 * 16
    out = [str(tmp_path / f"r{k}.vtk") for k in range(2)]
    assert hsd_main(["-s", *s, "-p", *p, "--icosahedron", "4", "--noguess", "-d", "2", "-c", *cf, "-o", *out]) == 0
    assert read_vtk(out[0]).vertices.shape[0] == 2562
