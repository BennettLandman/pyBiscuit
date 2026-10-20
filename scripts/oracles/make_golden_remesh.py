#!/usr/bin/env python
"""Differential-test matrix + golden generator for SphericalRemesh (plan row B4.2).

    .venv/bin/python scripts/oracles/make_golden_remesh.py --list
    .venv/bin/python scripts/oracles/make_golden_remesh.py <case> [<case> ...|--all]

Each case in :data:`CASES` is a *spec* (inputs are regenerated from seeds, never
committed) plus the oracle command line.  ``generate()`` writes the inputs into a
work directory; the oracle (``SphericalRemesh`` built by
``scripts/oracles/build_spherical_remesh.sh``) and the Python port
(``pybiscuit.sphere.remesh_cli.main``) are run on identical inputs and the
outputs compared (``tests/test_remesh.py``).  Cases flagged ``"golden": True``
are committed under ``tests/data/golden/spherical-remesh/<case>/`` (the files
the oracle wrote + ``command.json``) so the port is tested without the binary.

Spec keys
---------
sphere / ref : ``("ico", level)`` or ``("native", level, sigma, seed)``
    ``-s`` and ``-r`` meshes (``ref`` may be ``None`` = flag omitted).  ``native`` is a
    seeded Gaussian perturbation (``tests/fixtures/meshes.perturb``) of the icosphere,
    i.e. a non-unit "native-like" sphere.
subject : bool           write ``subject.vtk`` (``-s`` sphere with a smooth radial bump)
coeff : dict | None      ``{"deg": D, "pole": (az, elev), "seed": s, "scale": a}``
props : [(file, kind)]   per-vertex text files on the ``-s`` mesh; kind ``cont`` | ``label``
color : "ref" | None     ``color.txt``: one ``r g b`` int triple per *reference* vertex
argv : [str]             arguments after the program name (relative file names)
outputs : [str]          oracle outputs to keep; globs allowed
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ORACLES = Path.home() / "dev" / "biscuit" / "oracles"
BIN = ORACLES / "spherical-remesh" / "bin" / "SphericalRemesh"  # arm64 default (FMA fused)
# goldens come from the -ffp-contract=off build (x86-64-like float32 arithmetic of the SIF image);
# build with scripts/oracles/build_spherical_remesh_nofma.sh
BIN_NOFMA = ORACLES / "spherical-remesh-nofma" / "bin" / "SphericalRemesh"
GOLDEN = REPO / "tests" / "data" / "golden" / "spherical-remesh"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

SEED = 20261020

Q = "--quiet"
PROPS2 = [("lh.curv.txt", "cont"), ("lh.sulc.txt", "cont")]
COEFF6 = {"deg": 6, "pole": (0.7, 0.4), "seed": 3, "scale": 0.03}
COEFF3 = {"deg": 3, "pole": (-1.9, -0.6), "seed": 5, "scale": 0.05}


def _case(name, sphere, ref, argv, **kw):
    spec = {
        "sphere": sphere,
        "ref": ref,
        "subject": False,
        "coeff": None,
        "props": [],
        "color": None,
        "argv": argv,
        "outputs": [],
        "golden": False,
    }
    spec.update(kw)
    return name, spec


CASES: dict[str, dict] = dict(
    [
        # ---------------------------------------------------------------- property resampling
        _case(
            "prop_down",
            ("ico", 5),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-p",
                "lh.curv.txt",
                "lh.sulc.txt",
                "--outputProperty",
                "out",
                "--bary",
                "bary.txt",
                Q,
            ],
            props=PROPS2,
            outputs=["out.curv.txt", "out.sulc.txt", "bary.txt"],
            golden=True,
        ),
        _case(
            "prop_up",
            ("ico", 3),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-p",
                "lh.curv.txt",
                "lh.sulc.txt",
                "--outputProperty",
                "out",
                "--bary",
                "bary.txt",
                Q,
            ],
            props=PROPS2,
            outputs=["out.curv.txt", "out.sulc.txt", "bary.txt"],
        ),
        _case(
            "prop_up_ico6",
            ("ico", 5),
            ("ico", 6),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-p",
                "lh.curv.txt",
                "lh.sulc.txt",
                "--outputProperty",
                "out",
                "--bary",
                "bary.txt",
                Q,
            ],
            props=PROPS2,
            outputs=["out.curv.txt", "out.sulc.txt", "bary.txt"],
        ),
        # ---------------------------------------------------------------- parcellation call
        # reference/ctf_plain/SurfPostproc/common/parc/test.py:
        #   SphericalRemesh -p <flist> -r <native sphere> -s <ico sphere>
        #                   --outputProperty <data_folder>/<export_file> --quiet
        # -s = icosahedral sphere (network output lives there), -r = native sphere; the
        # outputs "<data_folder>/out.mat.<k>.txt" overwrite the inputs of the same name.
        _case(
            "parcellation",
            ("ico", 4),
            ("native", 4, 0.004, 11),
            [
                "-p",
                "data/out.mat.1.txt",
                "data/out.mat.2.txt",
                "data/out.mat.3.txt",
                "-r",
                "ref.vtk",
                "-s",
                "sphere.vtk",
                "--outputProperty",
                "data/out.mat",
                Q,
            ],
            props=[
                ("data/out.mat.1.txt", "cont"),
                ("data/out.mat.2.txt", "cont"),
                ("data/out.mat.3.txt", "cont"),
            ],
            outputs=["data/out.mat.1.txt", "data/out.mat.2.txt", "data/out.mat.3.txt"],
            golden=True,
        ),
        # ---------------------------------------------------------------- native spheres
        _case(
            "native_to_ico",
            ("native", 5, 0.004, 1),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-p",
                "lh.curv.txt",
                "lh.sulc.txt",
                "--outputProperty",
                "out",
                "--bary",
                "bary.txt",
                Q,
            ],
            props=PROPS2,
            outputs=["out.curv.txt", "out.sulc.txt", "bary.txt"],
            golden=True,
        ),
        _case(
            "ico_to_native",
            ("ico", 4),
            ("native", 5, 0.004, 2),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-p",
                "lh.curv.txt",
                "lh.sulc.txt",
                "--outputProperty",
                "out",
                "--bary",
                "bary.txt",
                Q,
            ],
            props=PROPS2,
            outputs=["out.curv.txt", "out.sulc.txt", "bary.txt"],
        ),
        # ---------------------------------------------------------------- nearest neighbour
        _case(
            "nneighbor",
            ("ico", 5),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-p",
                "lh.label.txt",
                "lh.curv.txt",
                "--nneighbor",
                "--outputProperty",
                "out",
                "--bary",
                "bary.txt",
                Q,
            ],
            props=[("lh.label.txt", "label"), ("lh.curv.txt", "cont")],
            outputs=["out.label.txt", "out.curv.txt", "bary.txt"],
            golden=True,
        ),
        _case(
            "nneighbor_surface",
            ("ico", 4),
            ("ico", 3),
            [
                "-s",
                "sphere.vtk",
                "-i",
                "subject.vtk",
                "-r",
                "ref.vtk",
                "-p",
                "lh.label.txt",
                "--nneighbor",
                "-o",
                "out.vtk",
                Q,
            ],
            subject=True,
            props=[("lh.label.txt", "label")],
            outputs=["out.vtk"],
        ),
        # ---------------------------------------------------------------- surface output
        _case(
            "surface",
            ("ico", 4),
            ("ico", 3),
            [
                "-s",
                "sphere.vtk",
                "-i",
                "subject.vtk",
                "-r",
                "ref.vtk",
                "-p",
                "lh.curv.txt",
                "lh.sulc.txt",
                "-o",
                "out.vtk",
                "--outputProperty",
                "out",
                Q,
            ],
            subject=True,
            props=PROPS2,
            outputs=["out.vtk", "out.curv.txt", "out.sulc.txt"],
            golden=True,
        ),
        _case(
            "surface_only",
            ("ico", 4),
            ("ico", 5),
            ["-s", "sphere.vtk", "-i", "subject.vtk", "-r", "ref.vtk", "-o", "out.vtk", Q],
            subject=True,
            outputs=["out.vtk"],
        ),
        _case(
            "surface_ref_only",
            ("ico", 3),
            ("ico", 4),
            ["-s", "sphere.vtk", "-r", "ref.vtk", "-o", "out.vtk", Q],
            outputs=["out.vtk"],
        ),
        # ---------------------------------------------------------------- colour
        _case(
            "color_nokeep_noinput",
            ("ico", 3),
            ("ico", 4),
            ["-s", "sphere.vtk", "-r", "ref.vtk", "--color", "color.txt", "-o", "out.vtk", Q],
            color="ref",
            outputs=["out.vtk"],
            golden=True,
        ),
        _case(
            "color_nokeep_input",
            ("ico", 3),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-i",
                "subject.vtk",
                "-r",
                "ref.vtk",
                "--color",
                "color.txt",
                "-o",
                "out.vtk",
                Q,
            ],
            subject=True,
            color="ref",
            outputs=["out.vtk"],
        ),
        _case(
            "color_keep",
            ("ico", 3),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-i",
                "subject.vtk",
                "-r",
                "ref.vtk",
                "--color",
                "color.txt",
                "--keepColor",
                "-p",
                "lh.curv.txt",
                "-o",
                "out.vtk",
                Q,
            ],
            subject=True,
            color="ref",
            props=[("lh.curv.txt", "cont")],
            outputs=["out.vtk"],
            golden=True,
        ),
        _case(
            "color_keep_nneighbor",
            ("ico", 3),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-i",
                "subject.vtk",
                "-r",
                "ref.vtk",
                "--color",
                "color.txt",
                "--keepColor",
                "--nneighbor",
                "-o",
                "out.vtk",
                Q,
            ],
            subject=True,
            color="ref",
            outputs=["out.vtk"],
        ),
        # ---------------------------------------------------------------- SH deformation
        _case(
            "coeff_full",
            ("native", 4, 0.004, 3),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-c",
                "coeff.txt",
                "-p",
                "lh.curv.txt",
                "--outputProperty",
                "out",
                "--deform",
                "deformed.vtk",
                "--bary",
                "bary.txt",
                Q,
            ],
            coeff=COEFF6,
            props=[("lh.curv.txt", "cont")],
            outputs=["out.curv.txt", "deformed.vtk", "bary.txt"],
            golden=True,
        ),
        _case(
            "coeff_deg2",
            ("ico", 5),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-c",
                "coeff.txt",
                "--deg",
                "2",
                "-p",
                "lh.curv.txt",
                "--outputProperty",
                "out",
                "--deform",
                "deformed.vtk",
                Q,
            ],
            coeff=COEFF6,
            props=[("lh.curv.txt", "cont")],
            outputs=["out.curv.txt", "deformed.vtk"],
        ),
        _case(
            "coeff_deg0",
            ("ico", 4),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-c",
                "coeff.txt",
                "--deg",
                "0",
                "-p",
                "lh.curv.txt",
                "--outputProperty",
                "out",
                "--deform",
                "deformed.vtk",
                Q,
            ],
            coeff=COEFF6,
            props=[("lh.curv.txt", "cont")],
            outputs=["out.curv.txt", "deformed.vtk"],
        ),
        _case(
            "coeff_sweep",
            ("ico", 4),
            ("ico", 3),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-c",
                "coeff.txt",
                "-p",
                "lh.curv.txt",
                "--outputProperty",
                "aug",
                "--deform",
                "def.vtk",
                "--bary",
                "bary.txt",
                "--deg0",
                "0",
                "--deg",
                "3",
                Q,
            ],
            coeff=COEFF6,
            props=[("lh.curv.txt", "cont")],
            outputs=["aug*.curv.txt", "def.vtk*", "bary.txt*"],
            golden=True,
        ),
        _case(
            "coeff_sweep_mid",
            ("ico", 4),
            ("ico", 3),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-c",
                "coeff.txt",
                "-p",
                "lh.curv.txt",
                "--outputProperty",
                "aug",
                "--deg0",
                "2",
                "--deg",
                "4",
                Q,
            ],
            coeff=COEFF6,
            props=[("lh.curv.txt", "cont")],
            outputs=["aug*.curv.txt"],
        ),
        _case(
            "coeff_deg0_gt_deg",
            ("ico", 4),
            ("ico", 3),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-c",
                "coeff.txt",
                "-p",
                "lh.curv.txt",
                "--outputProperty",
                "aug",
                "--deg0",
                "5",
                "--deg",
                "2",
                Q,
            ],
            coeff=COEFF6,
            props=[("lh.curv.txt", "cont")],
            outputs=["aug*.curv.txt"],
        ),
        _case(
            "coeff_deg0_only",
            ("ico", 4),
            ("ico", 3),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-c",
                "coeff.txt",
                "-p",
                "lh.curv.txt",
                "--outputProperty",
                "aug",
                "--deg0",
                "3",
                Q,
            ],
            coeff=COEFF3,
            props=[("lh.curv.txt", "cont")],
            outputs=["aug*.curv.txt"],
        ),
        _case(
            "coeff_nneighbor",
            ("ico", 5),
            ("ico", 4),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-c",
                "coeff.txt",
                "-p",
                "lh.label.txt",
                "--nneighbor",
                "--outputProperty",
                "out",
                Q,
            ],
            coeff=COEFF6,
            props=[("lh.label.txt", "label")],
            outputs=["out.label.txt"],
        ),
        _case(
            "coeff_surface",
            ("ico", 4),
            ("ico", 3),
            [
                "-s",
                "sphere.vtk",
                "-i",
                "subject.vtk",
                "-r",
                "ref.vtk",
                "-c",
                "coeff.txt",
                "-o",
                "out.vtk",
                "--deform",
                "deformed.vtk",
                Q,
            ],
            subject=True,
            coeff=COEFF3,
            outputs=["out.vtk", "deformed.vtk"],
        ),
        _case(
            "coeff_deform_only",
            ("ico", 4),
            None,
            ["-s", "sphere.vtk", "-c", "coeff.txt", "--deform", "deformed.vtk", Q],
            coeff=COEFF6,
            outputs=["deformed.vtk"],
        ),
        _case(
            "coeff_no_ref_prop",
            ("native", 4, 0.004, 7),
            None,
            [
                "-s",
                "sphere.vtk",
                "-c",
                "coeff.txt",
                "-p",
                "lh.curv.txt",
                "--outputProperty",
                "out",
                Q,
            ],
            coeff=COEFF6,
            props=[("lh.curv.txt", "cont")],
            outputs=["out.curv.txt"],
        ),
        # ---------------------------------------------------------------- binary types
        *[
            _case(
                f"ftype_{tag}",
                ("ico", 4),
                ("ico", 3),
                [
                    "-s",
                    "sphere.vtk",
                    "-r",
                    "ref.vtk",
                    "-p",
                    "lh.curv.txt",
                    "lh.label.txt",
                    "--ftype",
                    ft,
                    "--outputProperty",
                    "out",
                    Q,
                ],
                props=[("lh.curv.txt", "cont"), ("lh.label.txt", "label")],
                outputs=["out.curv.dat", "out.label.dat"],
                golden=tag in ("float", "int16"),
            )
            for tag, ft in [
                ("float", "float"),
                ("double", "double"),
                ("int", "int"),
                ("int16", "int16"),
                ("int32", "int32"),
                ("Float", "Float"),
                ("INT_16", "INT_16"),
                ("Double", "DOUBLE"),
            ]
        ],
        # ---------------------------------------------------------------- property names
        _case(
            "names",
            ("ico", 3),
            ("ico", 3),
            [
                "-s",
                "sphere.vtk",
                "-r",
                "ref.vtk",
                "-p",
                "lh_thick.txt",
                "a.b.c.mean.txt",
                "plain.txt",
                "--outputProperty",
                "o",
                "-o",
                "o.vtk",
                Q,
            ],
            props=[("lh_thick.txt", "cont"), ("a.b.c.mean.txt", "cont"), ("plain.txt", "cont")],
            outputs=["o.thick.txt", "o.mean.txt", "o.plain.txt", "o.vtk"],
        ),
    ]
)


# --------------------------------------------------------------------------- inputs
def _sphere(spec):
    from fixtures.meshes import perturb
    from pybiscuit.sphere.icosphere import icosphere

    if spec[0] == "ico":
        return icosphere(spec[1])
    return perturb(icosphere(spec[1]), spec[2], spec[3])


def _prop_values(v: np.ndarray, kind: str, k: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + 100 + k)
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    if kind == "label":
        return np.floor(2.5 * (u[:, 0] + 1) + 1.5 * (u[:, 2] > 0.1)).astype(int)
    return (
        np.sin((k + 2) * u[:, 0]) + np.cos(2 * u[:, 1]) + u[:, 2] ** 2
        + 0.01 * rng.standard_normal(len(u))
    )  # fmt: skip


def generate(spec: dict, work: Path) -> None:
    """Write all inputs of ``spec`` into ``work`` (deterministic)."""
    from pybiscuit.io import write_vtk
    from pybiscuit.mesh import Mesh
    from pybiscuit.sphere.spharm import ShDeformation, write_sh_coefficients

    work.mkdir(parents=True, exist_ok=True)
    sph = _sphere(spec["sphere"])
    write_vtk(sph, work / "sphere.vtk", dtype="float")
    nref = sph.n_vertices
    if spec["ref"] is not None:
        ref = _sphere(spec["ref"])
        write_vtk(ref, work / "ref.vtk", dtype="float")
        nref = ref.n_vertices
    v = sph.vertices
    if spec["subject"]:
        u = v / np.linalg.norm(v, axis=1, keepdims=True)
        r = 1 + 0.2 * u[:, 0] * u[:, 1] + 0.1 * u[:, 2]
        write_vtk(Mesh(u * r[:, None], sph.faces), work / "subject.vtk", dtype="float")
    for k, (name, kind) in enumerate(spec["props"]):
        p = work / name
        p.parent.mkdir(parents=True, exist_ok=True)
        vals = _prop_values(v, kind, k)
        fmt = "%d\n" if kind == "label" else "%.6f\n"
        p.write_text("".join(fmt % x for x in vals.tolist()))
    if spec["coeff"]:
        c = spec["coeff"]
        n = (c["deg"] + 1) ** 2
        rng = np.random.default_rng(c["seed"])
        l_of = np.floor(np.sqrt(np.arange(n))).astype(int)
        coeff = rng.standard_normal((3, n)) * c["scale"] / (1 + l_of)[None, :]
        write_sh_coefficients(
            work / "coeff.txt", ShDeformation(c["deg"], np.array(c["pole"]), coeff)
        )
    if spec["color"]:
        rng = np.random.default_rng(SEED + 7)
        col = rng.integers(0, 256, size=(nref, 3))
        (work / "color.txt").write_text("".join(f"{r} {g} {b}\n" for r, g, b in col.tolist()))


def input_files(spec: dict) -> list[str]:
    names = ["sphere.vtk"]
    if spec["ref"] is not None:
        names.append("ref.vtk")
    if spec["subject"]:
        names.append("subject.vtk")
    names += [n for n, _ in spec["props"]]
    if spec["coeff"]:
        names.append("coeff.txt")
    if spec["color"]:
        names.append("color.txt")
    return names


def compiler_version() -> str:
    out = subprocess.run(["/usr/bin/clang++", "--version"], capture_output=True, text=True)
    return out.stdout.splitlines()[0] if out.stdout else "unknown"


def run_oracle(spec: dict, work: Path, binary: Path = BIN) -> subprocess.CompletedProcess:
    r = subprocess.run([str(binary), *spec["argv"]], cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"oracle failed ({r.returncode}): {r.stderr}\n{r.stdout}")
    return r


def collect_outputs(spec: dict, work: Path) -> list[str]:
    """Relative names of the files the spec's ``outputs`` patterns match in ``work``."""
    names: list[str] = []
    for pat in spec["outputs"]:
        names += sorted(str(p.relative_to(work)) for p in work.glob(pat) if p.is_file())
    return names


def make_golden(name: str, out_root: Path = GOLDEN) -> Path:
    spec = CASES[name]
    dest = out_root / name
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        generate(spec, work)
        run_oracle(spec, work, BIN_NOFMA)
        outs = collect_outputs(spec, work)
        if not outs:
            raise RuntimeError(f"{name}: no outputs matched {spec['outputs']}")
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for rel in outs:
            tgt = dest / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(work / rel, tgt)
    meta = {
        "tool": "spherical-remesh",
        "case": name,
        "argv": ["$ORACLES/spherical-remesh-nofma/bin/SphericalRemesh", *spec["argv"]],
        "commit": "b677c39",
        "meshlib_commit": "dcef807",
        "fp_contract": "off",
        "compiler": compiler_version(),
        "oracle": "upstream-macos-arm64",
        "inputs": {
            "generator": "scripts/oracles/make_golden_remesh.py:generate",
            "seed": SEED,
            "spec": json.loads(json.dumps({k: v for k, v in spec.items() if k != "argv"})),
            "files": input_files(spec),
        },
        "outputs": outs,
    }
    (dest / "command.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    if size > 1_000_000:
        print(f"WARNING: {dest} is {size} bytes (>1MB)", file=sys.stderr)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--all", action="store_true", help="all cases flagged golden")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not (a.cases or a.all):
        for n, s in CASES.items():
            print(f"{n:24s} golden={s['golden']}")
        return
    for n in [n for n, s in CASES.items() if s["golden"]] if a.all else a.cases:
        print(make_golden(n))


if __name__ == "__main__":
    main()
