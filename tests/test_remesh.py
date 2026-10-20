"""SphericalRemesh port (plan row B4.2): differential tests vs the upstream oracle + goldens.

* ``test_golden_*``      -- port vs committed oracle outputs (no binary needed).
* ``test_oracle_*``      -- port vs the freshly run oracle binary on the whole matrix of
  ``scripts/oracles/make_golden_remesh.py:CASES`` (skipped when the binary is absent).
* ``test_parcellation_call_*`` -- the BISCUIT parcellation call
  ``SphericalRemesh -p <list> -r <native sphere> -s <ico sphere> --outputProperty <prefix> --quiet``
  (``reference/ctf_plain/SurfPostproc/common/parc/test.py``).

Comparison: header/face text must be identical; numbers are compared tokenwise with
``|a-b| <= ATOL + RTOL*|b|`` (the text has 6 / 9 decimals so ATOL covers print rounding).
Rows whose closest face differs from the oracle (exact barycentric ties, e.g. a reference vertex
that coincides with a sphere vertex) are exempt from value comparison and are counted.
Set ``REMESH_STATS_OUT=<file.json>`` to dump the measured statistics.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
BIN = ORACLES / "spherical-remesh" / "bin" / "SphericalRemesh"  # default arm64 flags (FMA fused)
BIN_NOFMA = ORACLES / "spherical-remesh-nofma" / "bin" / "SphericalRemesh"  # -ffp-contract=off
VARIANTS = {"fma": BIN, "nofma": BIN_NOFMA}
GOLDEN = ROOT / "tests" / "data" / "golden" / "spherical-remesh"
REFERENCE = ROOT.parent / "reference"

RTOL = 1e-5
ATOL_TXT = 1.1e-6  # '%f' printing
ATOL_VTK = 1e-6  # '%.9f' printing + a few float32 ulps near |x| = 1


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MG = _load(ROOT / "scripts" / "oracles" / "make_golden_remesh.py", "make_golden_remesh")
CASES = MG.CASES
STATS: dict = {}


def run_port(argv: list[str], cwd: Path) -> int:
    from pybiscuit.sphere.remesh_cli import main

    old = Path.cwd()
    os.chdir(cwd)
    try:
        return main(list(argv))
    finally:
        os.chdir(old)


# --------------------------------------------------------------------------- comparison
def _ftype_of(argv: list[str]) -> str | None:
    return argv[argv.index("--ftype") + 1] if "--ftype" in argv else None


def _close(a: np.ndarray, b: np.ndarray, atol: float) -> np.ndarray:
    return np.abs(a - b) <= atol + RTOL * np.abs(b)


def _rows_numeric(a_lines: list[str], b_lines: list[str]) -> np.ndarray:
    """Parse equally-long numeric text lines into (rows, ncols) float64 arrays."""
    return (
        np.array([x.split() for x in a_lines], float),
        np.array([x.split() for x in b_lines], float),
    )


def parse_vtk(path: Path) -> dict:
    """Parse a SphericalRemesh output VTK (points, faces, COLOR_SCALARS, FIELD arrays)."""
    lines = Path(path).read_text().split("\n")
    assert lines[:4] == ["# vtk DataFile Version 3.0", "vtk output", "ASCII", "DATASET POLYDATA"]
    n = int(lines[4].split()[1])
    assert lines[4] == f"POINTS {n} float"
    pts = lines[5 : 5 + n]
    i = 5 + n
    m = int(lines[i].split()[1])
    assert lines[i] == f"POLYGONS {m} {4 * m}"
    faces = lines[i + 1 : i + 1 + m]
    i += 1 + m
    assert lines[i] == ""  # blank line after POLYGONS
    i += 1
    out = {"points": pts, "faces": faces, "color": None, "fields": [], "tail": []}
    if i < len(lines) and lines[i].startswith("POINT_DATA"):
        assert lines[i] == f"POINT_DATA {n}"
        i += 1
        if lines[i] == "COLOR_SCALARS colors 3":
            out["color"] = lines[i + 1 : i + 1 + n]
            i += 1 + n
        if i < len(lines) and lines[i].startswith("FIELD ScalarData"):
            k = int(lines[i].split()[2])
            i += 1
            for _ in range(k):
                name, one, cnt, typ = lines[i].split()
                assert (one, int(cnt), typ) == ("1", n, "float")
                out["fields"].append((name, lines[i + 1 : i + 1 + n]))
                i += 1 + n
                assert lines[i] == ""
                i += 1
    out["tail"] = [x for x in lines[i:] if x != ""]
    assert out["tail"] == []
    return out


def compare_file(
    rel: str,
    ref: Path,
    got: Path,
    argv: list[str],
    ties: np.ndarray | None,
    trunc_slack: bool = False,
    bary_atol: float = 1e-5,
    val_rtol: float = RTOL,
) -> dict:
    """Compare ``got`` (port) against ``ref`` (oracle). Returns statistics; asserts structure.

    ``trunc_slack`` (FMA-fused oracle only): outputs produced by ``(int)`` truncation of an
    interpolated value (colours, ``--ftype int*``) may be off by one level on vertices where a
    reference vertex coincides with a sphere vertex (``1 - 1e-7`` vs ``1``); they are checked to
    be within one level but are not counted as mismatching rows.
    """
    a, b = Path(ref).read_bytes(), Path(got).read_bytes()
    st = {"bytes_equal": a == b, "size": len(a), "n_bad_rows": 0, "max_abs": 0.0, "max_rel": 0.0}
    bad_rows: set[int] = set()

    def num(x_ref, x_got, atol, rows_offset=0):
        r, g = np.asarray(x_ref, float), np.asarray(x_got, float)
        assert r.shape == g.shape, f"{rel}: shape {r.shape} vs {g.shape}"
        ok = np.abs(g - r) <= atol + val_rtol * np.abs(r)
        d = np.abs(g - r)
        if d.size:
            st["max_abs"] = max(st["max_abs"], float(np.nanmax(d)))
            nz = np.abs(r) > 1e-3
            if nz.any():
                st["max_rel"] = max(st["max_rel"], float(np.nanmax(d[nz] / np.abs(r[nz]))))
        bad = ~ok if ok.ndim == 1 else ~ok.all(axis=1)
        bad_rows.update(np.nonzero(bad)[0].tolist())

    if rel.endswith(".vtk") or ".vtk" in Path(rel).name:
        pa, pb = parse_vtk(ref), parse_vtk(got)
        assert pa["faces"] == pb["faces"], f"{rel}: face lines differ"
        assert (pa["color"] is None) == (pb["color"] is None), f"{rel}: color presence differs"
        assert [n for n, _ in pa["fields"]] == [n for n, _ in pb["fields"]], f"{rel}: field names"
        ra, rb = _rows_numeric(pa["points"], pb["points"])
        num(ra, rb, ATOL_VTK)
        if pa["color"] is not None:
            ca, cb = _rows_numeric(pa["color"], pb["color"])
            if trunc_slack:
                assert np.abs(ca - cb).max() <= 1.01 / 255
                st["color_rows_off_by_one"] = int((np.abs(ca - cb).max(axis=1) > 1e-5).sum())
            else:
                num(ca, cb, 1.1e-6)
        for (_, fa), (_, fb) in zip(pa["fields"], pb["fields"], strict=True):
            fa_, fb_ = _rows_numeric(fa, fb)
            num(fa_, fb_, ATOL_TXT)
    elif rel.endswith(".dat"):
        ft = _ftype_of(argv)
        dt = {"float": "<f4", "double": "<f8", "int16": "<i2", "int": "<i4", "int32": "<i4"}.get(ft)
        if dt is None:  # 'Float', 'INT_16', ...: validated by CLI11 but never written
            assert a == b == b""
            return st
        va, vb = np.frombuffer(a, dt), np.frombuffer(b, dt)
        assert va.shape == vb.shape
        if dt[1] == "i":  # (int) truncation of a float that sits next to an integer
            d = np.abs(va.astype(np.int64) - vb.astype(np.int64))
            assert d.max(initial=0) <= 1, f"{rel}: integer outputs differ by more than 1"
            if trunc_slack:
                st["int_off_by_one"] = int((d > 0).sum())
            else:
                bad_rows.update(np.nonzero(d > 0)[0].tolist())
            st["max_abs"] = float(d.max(initial=0))
        else:
            num(va.astype(float), vb.astype(float), 5e-7 if dt == "<f4" else 1e-6)
    elif Path(rel).name.startswith(("bary", "_tie")):
        la, lb = a.decode().split("\n"), b.decode().split("\n")
        assert len(la) == len(lb)
        ra = np.array([x.split() for x in la if x], float)
        rb = np.array([x.split() for x in lb if x], float)
        face_diff = ra[:, 0] != rb[:, 0]
        st["ties"] = int(face_diff.sum())
        same = ~face_diff
        ok = _close(
            rb[:, 1:], ra[:, 1:], bary_atol
        )  # coefficients are tiny differences + 6 decimals
        d = np.abs(rb[:, 1:] - ra[:, 1:])[same]
        st["max_abs"] = float(d.max(initial=0))
        bad_rows.update(np.nonzero(same & ~ok.all(axis=1))[0].tolist())
    else:  # per-vertex text list
        va = np.array(a.split(), float)
        vb = np.array(b.split(), float)
        assert va.shape == vb.shape, f"{rel}: {va.shape} vs {vb.shape}"
        num(va, vb, ATOL_TXT)

    if ties is not None:
        tie_set = set(np.nonzero(ties)[0].tolist())
        unexplained = bad_rows - tie_set
        st["n_bad_rows"] = len(bad_rows)
        st["n_unexplained"] = len(unexplained)
        assert not unexplained, f"{rel}: {len(unexplained)} out-of-tolerance rows not ties"
    else:
        st["n_bad_rows"] = len(bad_rows)
        st["n_unexplained"] = len(bad_rows)
        assert len(bad_rows) <= 2, f"{rel}: {len(bad_rows)} out-of-tolerance rows"
    return st


def tie_rows(oracle_dir: Path, port_dir: Path, bary_name: str) -> np.ndarray | None:
    """Boolean mask of reference vertices whose closest face differs (all suffixes)."""
    masks = []
    for p in sorted(oracle_dir.glob(bary_name + "*")):
        q = port_dir / p.name
        if not q.exists():
            continue
        ra = np.loadtxt(p, usecols=0, dtype=np.int64)
        rb = np.loadtxt(q, usecols=0, dtype=np.int64)
        masks.append(ra != rb)
    return np.any(masks, axis=0) if masks else None


# --------------------------------------------------------------------------- oracle matrix
needs_oracle = pytest.mark.skipif(not BIN.exists(), reason="SphericalRemesh oracle not built")


def _with_tie_bary(spec: dict) -> tuple[list[str], str]:
    argv = list(spec["argv"])
    if "--bary" in argv:
        return argv, argv[argv.index("--bary") + 1]
    return argv + ["--bary", "_tie_bary.txt"], "_tie_bary.txt"


def _variants():
    return [
        pytest.param(
            v, marks=pytest.mark.skipif(not b.exists(), reason=f"{v} SphericalRemesh oracle absent")
        )
        for v, b in VARIANTS.items()
    ]


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("name", list(CASES))
def test_oracle_matrix(name, variant, tmp_path):
    spec = CASES[name]
    wo, wp = tmp_path / "oracle", tmp_path / "port"
    MG.generate(spec, wo)
    MG.generate(spec, wp)
    argv, bary_name = _with_tie_bary(spec)
    assert MG.run_oracle({**spec, "argv": argv}, wo, VARIANTS[variant]).returncode == 0
    t0 = time.perf_counter()
    assert run_port(argv, wp) == 0
    dt = time.perf_counter() - t0
    ties = tie_rows(wo, wp, bary_name)
    outs = MG.collect_outputs({**spec, "outputs": spec["outputs"] + ["_tie_bary.txt*"]}, wo)
    assert outs
    got_outs = MG.collect_outputs({**spec, "outputs": spec["outputs"] + ["_tie_bary.txt*"]}, wp)
    assert outs == got_outs
    case_stats = {}
    for rel in outs:
        case_stats[rel] = compare_file(
            rel, wo / rel, wp / rel, argv, ties, trunc_slack=variant == "fma"
        )
    case_stats["_seconds_port"] = dt
    case_stats["_tie_rows"] = int(ties.sum()) if ties is not None else None
    STATS[f"{variant}/{name}"] = case_stats


@pytest.fixture(scope="session", autouse=True)
def _dump_stats():
    yield
    out = os.environ.get("REMESH_STATS_OUT")
    if out and STATS:
        Path(out).write_text(json.dumps(STATS, indent=1, default=str))


@needs_oracle
def test_oracle_bary_ties_are_geometric_ties(tmp_path):
    """Where port and oracle pick different faces, both reconstruct the same 3-D point."""
    spec = CASES["coeff_full"]
    wo, wp = tmp_path / "o", tmp_path / "p"
    for w in (wo, wp):
        MG.generate(spec, w)
    MG.run_oracle(spec, wo)
    assert run_port(spec["argv"], wp) == 0
    _check_tie_geometry(wo, wp)


def _check_tie_geometry(wo: Path, wp: Path) -> int:
    sph = np.array([x.split() for x in parse_vtk(wo / "deformed.vtk")["points"]], dtype=np.float64)
    faces = np.array([x.split()[1:] for x in parse_vtk(wo / "deformed.vtk")["faces"]], int)
    bo = np.loadtxt(wo / "bary.txt")
    bp = np.loadtxt(wp / "bary.txt")
    diff = bo[:, 0] != bp[:, 0]
    po = np.einsum("ij,ijk->ik", bo[:, 1:], sph[faces[bo[:, 0].astype(int)]])
    pp = np.einsum("ij,ijk->ik", bp[:, 1:], sph[faces[bp[:, 0].astype(int)]])
    assert np.abs(po - pp)[diff].max(initial=0) < 1e-5
    return int(diff.sum())


# --------------------------------------------------------------------------- committed goldens
GOLDEN_CASES = sorted(p.name for p in GOLDEN.glob("*") if (p / "command.json").exists())


def _run_golden(name: str, tmp_path: Path):
    meta = json.loads((GOLDEN / name / "command.json").read_text())
    if "generator" in meta["inputs"] and "make_golden_remesh" not in meta["inputs"]["generator"]:
        pytest.skip("O1a golden (separate test)")
    spec = CASES[name]
    MG.generate(spec, tmp_path)
    argv = meta["argv"][1:]
    assert argv == spec["argv"]
    assert run_port(argv, tmp_path) == 0
    stats = {}
    # a bary file in the golden gives the legitimate-tie mask for the other outputs
    bary = next((o for o in meta["outputs"] if Path(o).name.startswith("bary")), None)
    ties = None
    if bary:
        ra = np.loadtxt(GOLDEN / name / bary, usecols=0, dtype=np.int64)
        rb = np.loadtxt(tmp_path / bary, usecols=0, dtype=np.int64)
        ties = ra != rb
    for rel in meta["outputs"]:
        stats[rel] = compare_file(rel, GOLDEN / name / rel, tmp_path / rel, argv, ties)
    return stats


@pytest.mark.parametrize("name", [n for n in GOLDEN_CASES if n in CASES])
def test_golden(name, tmp_path):
    meta = json.loads((GOLDEN / name / "command.json").read_text())
    assert meta["oracle"] == "upstream-macos-arm64"
    assert meta["commit"] == "b677c39"
    _run_golden(name, tmp_path)


def test_parcellation_call_golden(tmp_path):
    """``SphericalRemesh -p <list> -r <native> -s <ico> --outputProperty <prefix> --quiet``."""
    meta = json.loads((GOLDEN / "parcellation" / "command.json").read_text())
    argv = meta["argv"][1:]
    # argument order of parc/test.py: -p <flist> -r osphere -s isphere --outputProperty P --quiet
    assert argv[0] == "-p" and argv[argv.index("-r") - 1].endswith(".txt")
    assert argv[argv.index("-r") + 2] == "-s"
    assert argv[-1] == "--quiet" and argv[-3] == "--outputProperty"
    stats = _run_golden("parcellation", tmp_path)
    # the outputs overwrite the inputs of the same name, like in the BISCUIT pipeline
    for rel in meta["outputs"]:
        assert (tmp_path / rel).exists()
        assert stats[rel]["n_unexplained"] == 0


@pytest.mark.parametrize("name", ["down5to4", "up3to4"])
def test_golden_o1a(name, tmp_path):
    """The two goldens committed by the oracle agent (``make_golden.sr_fixture``)."""
    mg = _load(ROOT / "scripts" / "oracles" / "make_golden.py", "make_golden")
    meta = json.loads((GOLDEN / name / "command.json").read_text())
    mg.sr_fixture(tmp_path, *mg.SR_CASES[name])
    argv = meta["argv"][1:]
    assert run_port(argv, tmp_path) == 0
    ref_bary = np.loadtxt(GOLDEN / name / "bary.txt", usecols=0, dtype=np.int64)
    got_bary = np.loadtxt(tmp_path / "bary.txt", usecols=0, dtype=np.int64)
    ties = ref_bary != got_bary
    for rel in meta["outputs"]:
        compare_file(rel, GOLDEN / name / rel, tmp_path / rel, argv, ties)


def test_golden_coeff_full_ties_are_geometric(tmp_path):
    """Differing bary rows vs the golden are the same 3-D point on the deformed sphere."""
    spec = CASES["coeff_full"]
    MG.generate(spec, tmp_path)
    assert run_port(spec["argv"], tmp_path) == 0
    # reuse the check with the golden playing the oracle role
    _check_tie_geometry(GOLDEN / "coeff_full", tmp_path)


# --------------------------------------------------------------------------- port-only tests
def test_ftype_validation():
    from pybiscuit.sphere.remesh import check_ftype

    for ok in ("INT_16", "Float", "DOUBLE", "int32", "int", "i_nt"):
        assert check_ftype(ok) == ok  # validated case/underscore-insensitively, kept verbatim
    with pytest.raises(ValueError):
        check_ftype("uint8")


def test_flags_match_parse_args_h():
    """Every flag of wrapper/PARSE_ARGS.h exists in the port with the same spelling."""
    src = REFERENCE / "repos" / "SphericalRemesh" / "wrapper" / "PARSE_ARGS.h"
    if not src.exists():
        pytest.skip("reference/ not available")
    flags = set()
    for m in re.finditer(r'add_(?:option|flag)\("([^"]+)"', src.read_text()):
        flags.update(m.group(1).split(","))
    from pybiscuit.sphere.remesh_cli import build_parser

    have = {s for a in build_parser()._actions for s in a.option_strings} - {"-h", "--help"}
    assert flags == have
    assert flags == {
        "--quiet", "-s", "--sphere", "-i", "--input", "-c", "--coeff", "-r", "--ref", "-p",
        "--property", "--color", "-o", "--output", "--outputProperty", "--deform", "--bary",
        "--ftype", "--nneighbor", "--deg", "--deg0", "--keepColor",
    }  # fmt: skip


def test_property_names_follow_upstream():
    from pybiscuit.sphere import remesh as R

    assert R._property_name_file("dir.x/lh.curv.txt") == "curv"
    assert R._property_name_file("lh_thick.txt") == "thick"
    assert R._property_name_file("plain.txt") == "plain"
    assert R._property_name_file("a.b.c.mean.txt") == "mean"
    # saveDeformedSurface only looks for '.'
    assert R._property_name_surface("lh_thick.txt") == "lh_thick"
    assert R._property_name_surface("data/out.mat.3.txt") == "3"


def test_cli_errors(tmp_path, capsys):
    from pybiscuit.sphere.remesh_cli import main

    assert main(["-r", "x.vtk"]) == 106  # --sphere required
    assert main(["-s", str(tmp_path / "nope.vtk")]) == 105  # ExistingFile
    f = tmp_path / "s.vtk"
    f.write_text("x")
    assert main(["-s", str(f), "--ftype", "uint8"]) == 105
    assert main(["-s", str(f), "--deg", "-2"]) in (105, 114)
    assert main(["-s", str(f), "--bogus"]) == 109
    assert main(["--help"]) == 0
    assert "Spherical Remeshing Tool" in capsys.readouterr().out


def test_subcommand_registered(capsys):
    from pybiscuit.cli import main

    assert main(["spherical-remesh", "--help"]) == 0
    out = capsys.readouterr().out
    assert "--outputProperty" in out and "--keepColor" in out
    assert main(["spherical-remesh", "-r", "x"]) == 106


def test_verbose_messages(tmp_path, capsys):
    spec = CASES["prop_up"]
    MG.generate(spec, tmp_path)
    argv = [a for a in spec["argv"] if a != "--quiet"]
    assert run_port(argv, tmp_path) == 0
    out = capsys.readouterr().out.split("\n")
    assert out[0] == "Loading unit sphere information.."
    assert "Load properties.." in out and "\t2 found!" in out
    assert "Tree initialization.." in out and "Property transferring.." in out
    assert out.count("Write output surface model..") == 1


def test_identity_resample(tmp_path):
    """Remesh onto the same sphere without -c: values at vertices are unchanged."""
    from pybiscuit.sphere.icosphere import icosphere
    from pybiscuit.sphere.remesh import resample

    s = icosphere(4)
    v = s.vertices
    f = (np.sin(3 * v[:, 0]) + v[:, 2] ** 2).astype(np.float32)
    (out,) = resample(s, s, f)
    assert np.abs(out - f).max() < 1e-5


def test_python_api_matches_cli(tmp_path):
    from pybiscuit.io import read_vtk
    from pybiscuit.sphere.remesh import resample

    spec = CASES["coeff_full"]
    MG.generate(spec, tmp_path)
    assert run_port(spec["argv"], tmp_path) == 0
    vals = np.loadtxt(tmp_path / "lh.curv.txt")
    (out,) = resample(
        tmp_path / "sphere.vtk", tmp_path / "ref.vtk", vals.astype(np.float32),
        coeff=tmp_path / "coeff.txt",
    )  # fmt: skip
    ref = np.loadtxt(tmp_path / "out.curv.txt")
    assert np.abs(out - ref).max() < 1.1e-6
    assert read_vtk(tmp_path / "deformed.vtk").n_vertices == 2562


def test_sweep_file_names(tmp_path):
    spec = CASES["coeff_sweep"]
    MG.generate(spec, tmp_path)
    assert run_port(spec["argv"], tmp_path) == 0
    names = sorted(p.name for p in tmp_path.iterdir())
    for i in range(4):
        assert f"aug{i}.curv.txt" in names and f"def.vtk{i}" in names and f"bary.txt{i}" in names


def test_deg_without_coeff_is_degree_zero(tmp_path):
    from pybiscuit.sphere.remesh import SphericalRemeshing

    spec = CASES["prop_up"]
    MG.generate(spec, tmp_path)
    sr = SphericalRemeshing(None, tmp_path / "sphere.vtk", None, False, tmp_path / "ref.vtk",
                            deg=5, verbose=False)  # fmt: skip
    assert sr.degree() == 0


@pytest.mark.skipif(not BIN_NOFMA.exists(), reason="nofma oracle absent")
def test_oracle_ico7_deg20(tmp_path):
    """Full-size case: 163,842 vertices, degree-20 deformation, two properties.

    One float32 ulp of difference in a deformed vertex (SH sums, libm sin/cos) moves the
    barycentric coefficients of an ico7 face (edge ~0.007) by ~1e-5, hence the looser bary
    tolerance (3e-5) and 3e-5 relative on the resampled values.
    """
    spec = {
        **CASES["coeff_full"],
        "sphere": ("native", 7, 0.0005, 5),
        "ref": ("ico", 7),
        "coeff": {"deg": 20, "pole": (0.3, -0.5), "seed": 9, "scale": 0.01},
        "props": [("lh.curv.txt", "cont"), ("lh.sulc.txt", "cont")],
        "argv": ["-s", "sphere.vtk", "-r", "ref.vtk", "-c", "coeff.txt", "-p", "lh.curv.txt",
                 "lh.sulc.txt", "--outputProperty", "out", "--deform", "deformed.vtk",
                 "--bary", "bary.txt", "--quiet"],
        "outputs": ["out.curv.txt", "out.sulc.txt", "deformed.vtk", "bary.txt"],
    }  # fmt: skip
    wo, wp = tmp_path / "o", tmp_path / "p"
    MG.generate(spec, wo)
    MG.generate(spec, wp)
    MG.run_oracle(spec, wo, BIN_NOFMA)
    t0 = time.perf_counter()
    assert run_port(spec["argv"], wp) == 0
    dt = time.perf_counter() - t0
    ties = tie_rows(wo, wp, "bary.txt")
    st = {}
    for f in spec["outputs"]:
        st[f] = compare_file(f, wo / f, wp / f, spec["argv"], ties, bary_atol=3e-5, val_rtol=3e-5)
    st["_seconds_port"] = dt
    st["_tie_rows"] = int(ties.sum())
    STATS["nofma/ico7_deg20"] = st
