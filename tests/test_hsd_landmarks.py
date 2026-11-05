# ruff: noqa: E501
"""HSD landmark term, location features and BOBYQA pole search (upstream-disabled options).

Goldens: ``tests/data/golden/hsd/lm_*`` from ``oracles/hsd-lm-nofma`` (HSD f169595 with
``patches/hsd-landmarks`` uncommenting the options; ``scripts/oracles/make_golden_hsd.py``) and
``tests/data/golden/bobyqa`` (upstream ``src/bobyqa.h`` run as a black box,
``scripts/oracles/make_golden_bobyqa.py``).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from fixtures.hsd_cases import (
    LM_EVAL_CASES,
    LM_RUN_CASES,
    coeff_states,
    expand_lm_flags,
    write_landmark_case,
)
from pybiscuit.hsd.bobyqa import bobyqa
from pybiscuit.hsd.cli import UPSTREAM_DISABLED, _list_dir, _subject_names, _trim
from pybiscuit.hsd.cli import main as hsd_main
from pybiscuit.hsd.register import HSD
from pybiscuit.io import read_vtk

GOLD = Path(__file__).parent / "data" / "golden"
F = np.float32


def _g(case):
    return np.load(GOLD / "hsd" / case / "golden.npz")


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ------------------------------------------------------------------------ BOBYQA
def _bobyqa_fn(case):
    cf = lambda v: F(math.cos(float(v)))  # noqa: E731
    sf = lambda v: F(math.sin(float(v)))  # noqa: E731
    if case == "quad":
        return lambda x: (
            (x[0] - 0.3) * (x[0] - 0.3) + 3 * (x[1] + 0.2) * (x[1] + 0.2) + 0.5 * x[0] * x[1]
        )
    if case == "rosen":
        return lambda x: 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1 - x[0]) * (1 - x[0])
    if case == "bound":
        return lambda x: (x[0] - 5) * (x[0] - 5) + (x[1] + 0.5) * (x[1] + 0.5)
    if case == "cosf":

        def f(x):
            a, b = F(x[0]), F(x[1])
            return float(
                (a - F(0.7)) * (a - F(0.7))
                + (b + F(0.4)) * (b + F(0.4))
                + F(0.05) * cf(F(9) * a) * sf(F(7) * b)
            )

        return f
    lm = np.array(
        [[0.6, 0, 0.8], [0, 0.6, 0.8], [0.48, 0.36, 0.8], [-0.36, 0.48, 0.8], [0, 0, 1]], np.float32
    )

    def f(x):
        phi, th = F(0.3) + F(x[0]), F(1.2) + F(x[1])
        p = [cf(th) * cf(phi), cf(th) * sf(phi), sf(th)]
        cost = F(0)
        for q in lm:
            inner = min(max(p[0] * q[0] + p[1] * q[1] + p[2] * q[2], F(-1)), F(1))
            a = F(math.acos(float(inner)))
            cost += a * a
        return float(cost)

    return f


@pytest.mark.parametrize("case", ["quad", "rosen", "bound", "cosf", "sphere"])
@pytest.mark.parametrize("maxfun", [3, 5, 8, 20, 200])
def test_bobyqa_trace_matches_bobyqa_h(case, maxfun):
    """Every evaluated point and the result are bit-identical to upstream ``bobyqa.h`` (-ffp-contract=off)."""
    g = np.load(GOLD / "bobyqa" / "golden.npz")
    f, ev = _bobyqa_fn(case), []

    def w(x):
        v = f(x)
        ev.append([x[0], x[1], v])
        return v

    x0 = [-1.2, 1.0] if case == "rosen" else [0.0, 0.0]
    x, _, _, nf = bobyqa(w, x0, [-math.pi] * 2, [math.pi] * 2, 0.1, 1e-6, maxfun)
    assert nf == len(ev)
    assert np.array_equal(np.array(ev), g[f"{case}_{maxfun}_evals"])
    assert np.array_equal(x, g[f"{case}_{maxfun}_x"])


def test_bobyqa_converges_with_bounds():
    x, f, _, _ = bobyqa(
        lambda x: (x[0] - 5) ** 2 + (x[1] + 0.5) ** 2, [0, 0], [-1, -1], [1, 1], 0.1, 1e-8, 500
    )
    assert x[0] == 1.0 and abs(x[1] + 0.5) < 1e-6


# ------------------------------------------------------------------- pole search
@pytest.fixture(scope="module")
def lm_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("hsd_lm")
    write_landmark_case(d, 3)
    return d


def _files(d, n, loc):
    sph = [str(d / f"s{k}.sphere.vtk") for k in range(n)]
    prop = [[str(d / f"s{k}.curv.txt")] for k in range(n)]
    lm = [str(d / f"s{k}.lm.txt") for k in range(n)]
    surf = [str(d / f"s{k}.surf.vtk") for k in range(n)] if loc else None
    return sph, prop, lm, surf


@pytest.mark.parametrize("max_iter", [0, 3, 6, 20, 100])
def test_landmark_pole_bitwise(lm_dir, max_iter):
    """``initTangentPlane`` with landmarks: pole, tangent basis and initial landmark features."""
    g = _g("lm_pole")
    sph, prop, lm, _ = _files(lm_dir, 3, False)
    h = HSD(
        sph,
        prop,
        degree=6,
        icosahedron_level=3,
        guess=False,
        max_iter=max_iter,
        landmarks=lm,
        verbose=False,
    )
    for s, sub in enumerate(h.subj):
        assert np.array_equal(sub.pole, g[f"pole_{max_iter}_{s}"])
        assert np.array_equal(np.stack([sub.tan1, sub.tan2]), g[f"tan_{max_iter}_{s}"])
    assert np.array_equal(h.lm_x, g[f"landmark_{max_iter}"])


# -------------------------------------------------------------------- unit level
def _flag(flags, name, default):
    return float(flags[flags.index(name) + 1]) if name in flags else default


def _lm_eval(c, d):
    loc = "--locationWeight" in c["flags"]
    sph, prop, lm, surf = _files(d, 3, loc)
    h = HSD(
        sph, prop, degree=c["deg"], icosahedron_level=c["ico"], guess=False, landmarks=lm, surfaces=surf,
        weight_loc=_flag(c["flags"], "--locationWeight", 0.0), weight_map=_flag(c["flags"], "--weightMap", 1.0),
        verbose=False,
    )  # fmt: skip
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
    r = {
        "eA": [h.cost(), h.cost_prop, h.cost_area, h.cost_disp],
        "lmA": h.cost_lm,
        "landmark": h.lm_x.copy(),
    }
    r["vA"] = [s.v.copy() for s in h.subj]
    r["feat"] = (h.feature.copy(), h.mean.copy(), h.variance.copy())
    solve = h._update_new_gradient
    h._update_new_gradient = lambda *a: None  # raw (undamped) gradient first
    h.update_gradient(c["beg"], c["end"], 0.001)
    r["graw"], r["H"] = h.gradient.copy(), list(h.hessian)
    h._update_new_gradient = solve
    h.update_gradient(c["beg"], c["end"], 0.001)
    r["gdir"] = h.gradient.copy()
    h.coeff[:] = B
    r["eB"] = [h.cost(), h.cost_prop, h.cost_area, h.cost_disp]
    r["lmB"] = h.cost_lm
    r["h"] = h
    return r


@pytest.mark.parametrize("name", list(LM_EVAL_CASES))
def test_lm_unit_level_vs_oracle(name, lm_dir):
    c, g = LM_EVAL_CASES[name], _g(name)
    r = _lm_eval(c, lm_dir)
    h = r["h"]
    for s, sub in enumerate(h.subj):
        assert np.array_equal(sub.pole, g[f"pole_{s}"])
        assert np.array_equal(np.stack([sub.tan1, sub.tan2]), g[f"tan_{s}"])
        assert np.abs(r["vA"][s] - g[f"vertex_A_{s}"]).max() < 3e-7  # float32 ulp noise from libm
    assert np.abs(r["landmark"] - g["landmark_A"]).max() < 3e-7
    for key, gold in (("eA", g["energy_A"]), ("eB", g["energy_B"])):
        assert _rel(r[key], gold[:4]) < 2e-7, (name, key, r[key], gold[:4])
    assert not g["energy_A"][4:].any()
    assert _rel(r["lmA"], g["cost_lm_A"]) < 2e-7 and _rel(r["lmB"], g["cost_lm_B"]) < 2e-7
    fine = c["ico"] >= 6
    for s in range(3):
        assert _rel(r["graw"][s], g[f"gradient_raw_{s}"]) < 1e-5, (name, s)
        scale = np.abs(g[f"gradient_dir_{s}"]).max()
        assert np.abs(r["gdir"][s] - g[f"gradient_dir_{s}"]).max() < 2e-5 * scale, (name, s)
        if fine and f"hessian_{s}" in g:
            assert _rel(r["H"][s], g[f"hessian_{s}"]) < 1e-5, (name, s)
    if "feature_A" in g:
        assert np.abs(r["feat"][0] - g["feature_A"]).max() < 1e-6
        assert np.abs(r["feat"][1] - g["mean_A"]).max() < 1e-6


def test_location_features_are_three_extra_properties(lm_dir):
    sph, prop, lm, surf = _files(lm_dir, 3, True)
    h = HSD(
        sph,
        prop,
        degree=4,
        icosahedron_level=3,
        guess=False,
        surfaces=surf,
        weight_loc=0.7,
        verbose=False,
    )
    assert h.nprop == 4 and h.nsurf == 3 and h.nlm == 0 and h.eta == 1.0
    h0 = HSD(
        sph,
        prop,
        degree=4,
        icosahedron_level=3,
        guess=False,
        surfaces=surf,
        weight_loc=0.0,
        verbose=False,
    )
    assert h0.nprop == 1  # main.cpp: no location features unless --locationWeight > 0
    h1 = HSD(sph, prop, degree=4, icosahedron_level=3, guess=False, weight_map=0.3, verbose=False)
    assert h1.eta == 1.0  # --weightMap applies only with landmarks


# --------------------------------------------------------------------- full runs
def _angles(a, b):
    cs = np.clip((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1), -1, 1)
    return np.degrees(np.arccos(cs))


@pytest.mark.parametrize("name", list(LM_RUN_CASES))
def test_lm_full_run_vs_oracle(name, tmp_path, capfd, monkeypatch):
    c, g = LM_RUN_CASES[name], _g(name)
    nsub = c["nsub"]
    write_landmark_case(tmp_path, nsub)
    monkeypatch.chdir(tmp_path)
    argv = (
        ["-s"] + [f"s{k}.sphere.vtk" for k in range(nsub)]
        + ["-p"] + [f"s{k}.curv.txt" for k in range(nsub)]
        + ["-o"] + [f"p{k}.reg.vtk" for k in range(nsub)]
        + ["--outputcoeff"] + [f"p{k}.coeff" for k in range(nsub)]
        + expand_lm_flags(c["flags"], nsub, tmp_path)
    )  # fmt: skip
    assert hsd_main(argv) == 0
    log = capfd.readouterr().out
    en = np.array([float(m.group(1)) for m in re.finditer(r"^\[\d+\] ([0-9.e+-]+)", log, re.M)])
    gold = g["energies"]
    ens = "ens_E" in g  # damped Newton runs: the oracle's own 1-ulp spread is stored (Q15)
    bound = (
        (lambda pol, key, floor=1e-6: max(pol, 1.5 * float(np.max(g[key])) + floor))
        if ens
        else (lambda pol, key, floor=0: pol)
    )
    if ens:
        assert (
            abs(len(en) - len(gold))
            <= int(np.ceil(1.5 * np.abs(g["ens_iters"] - len(gold)).max())) + 2
        )
        e_spread = np.abs(g["ens_E"] - gold[-1]) / gold[-1]
        assert abs(en[-1] - gold[-1]) / gold[-1] <= max(1e-3, 1.5 * e_spread.max()), (
            name,
            en[-1],
            gold[-1],
        )
    else:
        assert len(en) == len(gold)
        assert abs(en[-1] - gold[-1]) / gold[-1] < 1e-3
    for k in range(nsub):
        if f"vertex_{k}" not in g:
            assert not Path(f"p{k}.reg.vtk").exists()
            continue
        ang = _angles(read_vtk(f"p{k}.reg.vtk").vertices, g[f"vertex_{k}"].astype(np.float64))
        assert np.median(ang) <= bound(0.01, "ens_ang_med"), (name, k, np.median(ang))
        assert np.percentile(ang, 99) <= bound(0.1, "ens_ang_p99"), (
            name,
            k,
            np.percentile(ang, 99),
        )
        toks = Path(f"p{k}.coeff").read_text().split()
        cf = np.array(toks[3:], dtype=np.float64).reshape(-1, 3).T
        assert np.abs(cf - g[f"coeff_{k}"]).max() <= bound(1e-4, "ens_coeff", 2e-6), (name, k)
        assert np.array_equal(np.array(toks[1:3], dtype=np.float64), g[f"pole_{k}"])


def test_landmarks_pull_curves_together(lm_dir):
    """Sanity independent of the oracle: the landmark energy decreases."""
    sph, prop, lm, _ = _files(lm_dir, 3, False)
    h = HSD(
        sph,
        prop,
        degree=4,
        icosahedron_level=4,
        guess=False,
        landmarks=lm,
        weight_map=0.1,
        verbose=False,
    )
    h.trace()
    before = h.cost_lm
    h.optimization()
    h.trace()
    assert h.cost_lm < 0.5 * before


# --------------------------------------------------------------------------- CLI
def test_dir_forms_match_main_cpp(tmp_path):
    write_landmark_case(tmp_path, 3)
    expand_lm_flags(["--landmarkDir", "lmdir"], 3, tmp_path)
    names = _subject_names([str(tmp_path / f"s{k}.sphere.vtk") for k in range(3)])
    assert names == ["s0", "s1", "s2"]
    lst = _trim(_list_dir(str(tmp_path / "lmdir"), "txt"), names)
    assert [Path(p).name for p in lst] == ["s0.lm.txt", "s1.lm.txt", "s2.lm.txt"]
    assert [Path(p).name for p in _list_dir(str(tmp_path / "surfdir"), "vtk")] == [
        f"s{k}.surf.vtk" for k in range(3)
    ]


def test_upstream_disabled_flags_are_commented_out_upstream():
    src = Path("~/dev/biscuit/reference/repos/HSD/wrapper/PARSE_ARGS.h").expanduser()
    if not src.exists():
        pytest.skip("reference/repos/HSD not available")
    text = src.read_text()
    disabled = set()
    for block in re.findall(r"/\*.*?\*/", text, flags=re.S):
        disabled |= {
            s for m in re.findall(r'add_(?:option|flag)\("([^"]+)"', block) for s in m.split(",")
        }
    for line in text.splitlines():
        if line.strip().startswith("//"):
            disabled |= {
                s for m in re.findall(r'add_(?:option|flag)\("([^"]+)"', line) for s in m.split(",")
            }
    assert set(UPSTREAM_DISABLED) <= disabled


def test_cli_errors_landmarks(tmp_path, capsys):
    write_landmark_case(tmp_path, 3)
    s = [str(tmp_path / f"s{k}.sphere.vtk") for k in range(3)]
    p = [str(tmp_path / f"s{k}.curv.txt") for k in range(3)]
    base = ["-s", *s, "-p", *p, "--outputcoeff", "a", "b", "c"]
    assert hsd_main(base + ["-l", str(tmp_path / "s0.lm.txt")]) == 1
    assert "landmark files" in capsys.readouterr().out
    assert (
        hsd_main(base + ["--surface", str(tmp_path / "s0.surf.vtk"), "--locationWeight", "1"]) == 1
    )
    assert "surface files" in capsys.readouterr().out
