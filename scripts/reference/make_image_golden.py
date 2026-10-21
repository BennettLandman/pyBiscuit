#!/usr/bin/env python
"""B1.4: replay the oracle golden cases inside ``biscuit_FC_v2.2.sif`` (Linux, Apptainer/Singularity).

    python scripts/reference/make_image_golden.py --sif biscuit_FC_v2.2.sif --list
    python scripts/reference/make_image_golden.py --sif IMG --all [--keep-going] [--out DIR]
    python scripts/reference/make_image_golden.py --sif IMG --tool hsd --case rot8_ico4
    python scripts/reference/make_image_golden.py --sif IMG --all --dry-run   # print the commands only

Writes ``tests/data/golden-image/<tool>/<case>/`` (same layout as ``tests/data/golden/``) with
``command.json`` carrying ``"oracle": "image-biscuit_FC_v2.2"`` and the image SHA-256.

Two sources of cases:

* every case registered in ``scripts/oracles/make_golden.py`` (``REGISTRY``): the fixture and
  argv are reused verbatim; only the oracle binary path is replaced by the image tool name and
  the process is run with ``apptainer exec`` (so cases other briefs register there are replayed too);
* image-only tools defined below (``MeshProperty``, ``MeshDeform``, ``ShapeComplexityIndex``,
  ``klaplace``, ``OuterHull`` and an LGI chain).  Their flags come from the binaries' strings
  (never run on the owner's Mac): ``help.txt`` (``<tool> --help``) is stored with every such case so
  wrong flags are visible, and ``--keep-going`` records failures instead of stopping.

Never run on macOS: the image is x86-64 Linux.  ``--dry-run`` works anywhere.  A fake image plus
an ``apptainer`` stub is used by ``tests/test_reference_harness.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "tests" / "data" / "golden-image"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "oracles"))

import make_golden as mg  # noqa: E402

ORACLE_TAG = "image-biscuit_FC_v2.2"
EXPECTED_SHA256 = "e70aea6290ca3312a290618a8092b8992444bc47a7d3b42680109dfd714d4329"
IMAGE_TOOLS = {  # tool name in the image -> directory under /opt
    "SphericalRemesh": "cmorph", "HSD": "cmorph", "CurveExtraction": "cmorph",
    "Gyrification": "cmorph", "OuterHull": "cmorph", "klaplace": "cmorph",
    "MeshProperty": "biscuit", "MeshDeform": "biscuit", "MeshContour": "biscuit",
    "ShapeComplexityIndex": "biscuit",
}  # fmt: skip
LARGE = {("spherical-remesh", "up5to6")}  # >5 MB outputs: not committed (see make_golden.py)
GOLDEN_MAX_BYTES = 1_000_000


# ---------------------------------------------------------------- image execution
class Image:
    """``apptainer exec`` wrapper.  ``dry`` prints commands and skips execution."""

    def __init__(self, sif: Path, app: str, dry: bool = False):
        self.sif, self.app, self.dry = Path(sif), app, dry
        self.log: list[list[str]] = []

    def command(self, argv: list[str], work: Path) -> list[str]:
        tool = Path(argv[0]).name
        if tool not in IMAGE_TOOLS:
            raise RuntimeError(f"{argv[0]!r} is not an image tool ({sorted(IMAGE_TOOLS)})")
        args = [str(a).replace(str(work), "/work") for a in argv[1:]]
        return [
            self.app, "exec", "--cleanenv", "--pwd", "/work", "-B", f"{work}:/work",
            "--env", "MCR_CACHE_ROOT=/work/.mcr", "--env", "OMP_NUM_THREADS=1",
            str(self.sif), tool, *args,
        ]  # fmt: skip

    def version(self) -> str:
        if self.dry:
            return "dry-run"
        r = subprocess.run([self.app, "--version"], capture_output=True, text=True)
        return (r.stdout or r.stderr).strip()


class Session:
    """Routes ``make_golden.run`` into the image and records what was run."""

    def __init__(self, image: Image):
        self.image = image
        self.work: Path | None = None
        self.commands: list[list[str]] = []
        self.before: set[str] | None = None

    def snapshot(self) -> set[str]:
        assert self.work is not None
        return {
            str(p.relative_to(self.work))
            for p in self.work.rglob("*")
            if p.is_file() and ".mcr" not in p.parts
        }

    def run(self, argv, cwd):
        self.work = Path(cwd)
        if self.before is None:
            self.before = self.snapshot()  # inputs are complete at the first command
        argv = [str(a) for a in argv]
        self.commands.append(argv)
        cmd = self.image.command(argv, Path(cwd))
        print("+ " + " ".join(cmd), flush=True)
        if self.image.dry:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
        return r

    def new_files(self) -> list[str]:
        assert self.before is not None
        return sorted(self.snapshot() - self.before)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- image-only cases
IMAGE_REGISTRY: dict = {}


def image_case(tool: str, cases: list[str]):
    def deco(fn):
        fn.cases = cases
        IMAGE_REGISTRY[tool] = fn
        return fn

    return deco


def bumpy_fixture(work: Path, level: int = 5, name: str = "bumpy.vtk") -> dict:
    """Bumpy sphere (~50 mm) as in tests/test_oracle_smoke_curve.py, plus a convex-ish hull."""
    from pybiscuit.io import write_vtk
    from pybiscuit.mesh import Mesh
    from pybiscuit.sphere.icosphere import icosphere

    m = icosphere(level)
    u = m.vertices / np.linalg.norm(m.vertices, axis=1, keepdims=True)
    x, y, z = u.T
    r = 50 * (1 + 0.10 * np.sin(5 * x + 1) * np.cos(4 * y) + 0.07 * np.sin(6 * z + x * 3))
    write_vtk(Mesh(u * r[:, None], m.faces), work / name, dtype="float")
    write_vtk(Mesh(u * 55, m.faces), work / "hull.vtk", dtype="float")
    return {"generator": "make_image_golden.bumpy_fixture", "level": level, "files": [name, "hull.vtk"]}


def sh(tool: str, *args: str) -> list[str]:
    return [tool, *args]


@image_case("hsd", ["rot8_ico4"])
def hsd_case(case: str, work: Path, run) -> dict:
    from pybiscuit.io import write_vtk
    from pybiscuit.mesh import Mesh
    from pybiscuit.sphere.icosphere import icosphere

    m = icosphere(4)
    v = m.vertices
    u = v / np.linalg.norm(v, axis=1, keepdims=True)
    th = np.deg2rad(8.0)
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    write_vtk(Mesh(v.copy(), m.faces), work / "s1.sphere.vtk", dtype="float")
    write_vtk(Mesh(v @ rot.T, m.faces), work / "s2.sphere.vtk", dtype="float")
    x, y, z = u.T
    prop = np.sin(3 * x) + np.cos(2 * y) + 0.5 * z**2 + 0.7 * np.sin(2 * x + z)
    for n in ("s1", "s2"):
        np.savetxt(work / f"{n}.curv.txt", prop, fmt="%.9g")
    argv = sh("HSD", "-s", "s1.sphere.vtk", "s2.sphere.vtk", "-p", "s1.curv.txt", "s2.curv.txt",
              "-o", "s1.reg.vtk", "s2.reg.vtk", "-d", "6", "--icosahedron", "3", "--maxIter", "2")  # fmt: skip
    run(argv, work)
    return {"argv": argv, "commit": "f169595", "outputs": "auto",
            "inputs": {"generator": "make_image_golden.hsd_case", "rotation_deg": 8, "level": 4}}  # fmt: skip


@image_case("curve-extraction", ["bumpy_ico5"])
def curve_case(case: str, work: Path, run) -> dict:
    inputs = bumpy_fixture(work)
    argv = sh("CurveExtraction", "-i", "bumpy.vtk", "-o", "bumpy", "--sulcus", "--gyrus", "--cart",
              "--bary", "--interm", "--sulcalSeed", "-0.01", "--gyralSeed", "0.01")  # fmt: skip
    run(argv, work)
    return {"argv": argv, "commit": "ded932d", "outputs": "auto", "inputs": inputs}


@image_case("lgi-chain", ["bumpy_ico5"])
def lgi_chain_case(case: str, work: Path, run) -> dict:
    """The sub-tool calls of the image's ``lgi`` script (OuterHull, klaplace, CurveExtraction, Gyrification)."""
    inputs = bumpy_fixture(work)
    cmds = [
        sh("OuterHull", "bumpy.vtk", "bumpy.hull.vtk"),
        sh("klaplace", "-thread", "1", "-dims", "128", "bumpy.vtk", "bumpy.hull.vtk",
           "-surfaceCorrespondence", "bumpy.corr"),
        sh("klaplace", "-conv", "bumpy.corr_warpedMesh.vtp", "bumpy.hull.corr.vtk"),
        sh("CurveExtraction", "-i", "bumpy.vtk", "-o", "bumpy", "--sulcus", "--gyrus", "--bary",
           "--sulcalSeed", "-0.01", "--gyralSeed", "0.01"),
        sh("Gyrification", "-i", "bumpy.vtk", "-o", "lgi", "--outer", "bumpy.hull.corr.vtk",
           "-s", "bumpy.scurve.bary", "-g", "bumpy.gcurve.bary", "-m", "316", "--speed", "0.2",
           "--refHullArea", "77100", "--nThreads", "1"),
    ]  # fmt: skip
    for c in cmds:
        run(c, work)
    return {"argv": cmds[-1], "commands": cmds, "commit": "image", "outputs": "auto", "inputs": inputs}


@image_case("meshproperty", ["all_maps", "smoothed", "quad"])
def meshproperty_case(case: str, work: Path, run) -> dict:
    inputs = bumpy_fixture(work, level=4)
    flags = {
        "all_maps": ["-c", "-s", "-k", "-m", "-v", "--writevtk"],
        "smoothed": ["-m", "-k", "--tensorsmoothing", "2", "--surfsmoothing", "1",
                     "--datasmoothing", "2", "--writevtk"],
        "quad": ["-q", "-m", "-k", "--writevtk"],
    }[case]  # fmt: skip
    argv = sh("MeshProperty", "-i", "bumpy.vtk", "-o", "bumpy", *flags)
    run(argv, work)
    return {"argv": argv, "commit": "image", "outputs": "auto", "inputs": inputs, "help": ["MeshProperty", "--help"]}  # fmt: skip


@image_case("meshdeform", ["translate", "matrix", "sphere"])
def meshdeform_case(case: str, work: Path, run) -> dict:
    inputs = bumpy_fixture(work, level=4)
    flags = {
        "translate": ["-t", "1.5", "-2.0", "0.5"],
        "matrix": ["--matrix", "1", "0", "0", "2", "0", "1", "0", "-1", "0", "0", "1", "0.5"],
        "sphere": ["--sphere"],
    }[case]  # fmt: skip
    argv = sh("MeshDeform", "-i", "bumpy.vtk", "-o", "deformed.vtk", *flags)
    run(argv, work)
    return {"argv": argv, "commit": "image", "outputs": "auto", "inputs": inputs, "help": ["MeshDeform", "--help"]}  # fmt: skip


@image_case("shape-complexity-index", ["ico4_k6"])
def sci_case(case: str, work: Path, run) -> dict:
    inputs = bumpy_fixture(work, level=4)
    argv = sh("ShapeComplexityIndex", "--input", "bumpy.vtk", "--templatesurface", "bumpy.vtk",
              "--kerenlsize", "6", "--stepsize", "1", "--output", "sci.txt", "--format", "ASCII")  # fmt: skip
    run(argv, work)
    return {"argv": argv, "commit": "aea9afd", "outputs": "auto", "inputs": inputs, "help": ["ShapeComplexityIndex", "--help"]}  # fmt: skip


@image_case("klaplace", ["corr_ico4"])
def klaplace_case(case: str, work: Path, run) -> dict:
    inputs = bumpy_fixture(work, level=4)
    cmds = [
        sh("klaplace", "-thread", "1", "-dims", "128", "bumpy.vtk", "hull.vtk",
           "-surfaceCorrespondence", "corr"),
        sh("klaplace", "-conv", "corr_warpedMesh.vtp", "corr_warpedMesh.vtk"),
    ]  # fmt: skip
    for c in cmds:
        run(c, work)
    return {"argv": cmds[0], "commands": cmds, "commit": "93b0d02", "outputs": "auto", "inputs": inputs}


@image_case("outer-hull", ["bumpy_ico5"])
def outer_hull_case(case: str, work: Path, run) -> dict:
    inputs = bumpy_fixture(work)
    argv = sh("OuterHull", "bumpy.vtk", "bumpy.hull.vtk")
    run(argv, work)
    return {"argv": argv, "commit": "image", "outputs": "auto", "inputs": inputs}


# ---------------------------------------------------------------- driver
def all_cases(include_large: bool = False) -> list[tuple[str, str, str]]:
    """(tool, case, source) with source 'oracle-replay' or 'image-only'."""
    out = []
    for tool, fn in sorted(mg.REGISTRY.items()):
        for c in getattr(fn, "cases", []):
            if include_large or (tool, c) not in LARGE:
                out.append((tool, c, "oracle-replay"))
    for tool, fn in sorted(IMAGE_REGISTRY.items()):
        out.extend((tool, c, "image-only") for c in fn.cases)
    return out


def make_image_golden(tool: str, case: str, image: Image, out_root: Path = OUT_ROOT) -> Path | None:
    sha = EXPECTED_SHA256 if image.dry else sha256_file(image.sif)
    sess = Session(image)
    old_run = mg.run
    mg.run = sess.run
    try:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            if tool in IMAGE_REGISTRY:
                res = IMAGE_REGISTRY[tool](case, work, sess.run)
            else:
                res = mg.REGISTRY[tool](case, work, Path("/__oracles__"))
            if image.dry:
                return None
            outputs = res["outputs"]
            if outputs == "auto":
                outputs = sess.new_files()
            dest = out_root / tool / case
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            for rel in outputs:
                shutil.copy(work / rel, dest / Path(rel).name)
            if res.get("help"):
                h = subprocess.run(image.command(res["help"], work), capture_output=True, text=True)
                (dest / "help.txt").write_text(h.stdout + h.stderr)
            cmds = [[Path(c[0]).name, *c[1:]] for c in sess.commands]
            meta = {
                "tool": tool,
                "case": case,
                "argv": [Path(res["argv"][0]).name, *res["argv"][1:]],
                "commands": cmds,
                "commit": res["commit"],
                "oracle": ORACLE_TAG,
                "image": "biscuit_FC_v2.2.sif",
                "image_sha256": sha,
                "runtime": image.version(),
                "inputs": res.get("inputs", {}),
                "outputs": [Path(o).name for o in outputs],
            }
            (dest / "command.json").write_text(json.dumps(meta, indent=2) + "\n")
    finally:
        mg.run = old_run
    size = sum(p.stat().st_size for p in dest.iterdir())
    if size > GOLDEN_MAX_BYTES:
        print(f"WARNING: {dest} is {size} bytes (>1MB); do not commit", file=sys.stderr)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sif", type=Path, help="biscuit_FC_v2.2.sif")
    ap.add_argument("--apptainer", default=None, help="apptainer|singularity (default: autodetect)")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--tool")
    ap.add_argument("--case")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--include-large", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip the image SHA-256 check (mock tests)")
    a = ap.parse_args(argv)

    cases = all_cases(a.include_large)
    if a.list:
        for t, c, src in cases:
            print(f"{t}\t{c}\t{src}")
        return 0
    if not a.sif:
        ap.error("--sif is required")
    if a.all:
        sel = cases
    elif a.tool and a.case:
        sel = [x for x in all_cases(True) if x[0] == a.tool and x[1] == a.case]
        if not sel:
            ap.error(f"unknown case {a.tool}/{a.case}; try --list")
    else:
        ap.error("give --all or --tool and --case")
    app = a.apptainer or shutil.which("apptainer") and "apptainer" or shutil.which("singularity") and "singularity"
    if not app and not a.dry_run:
        raise SystemExit("error: neither apptainer nor singularity found (Linux host required)")
    app = app or "apptainer"
    if not a.dry_run:
        if not a.sif.is_file():
            raise SystemExit(f"error: image not found: {a.sif}")
        if not a.no_verify and sha256_file(a.sif) != EXPECTED_SHA256:
            raise SystemExit(f"error: {a.sif} sha256 != {EXPECTED_SHA256}")
    image = Image(a.sif, app, dry=a.dry_run)
    failed = []
    for tool, case, _src in sel:
        print(f"== {tool}/{case}", flush=True)
        try:
            dest = make_image_golden(tool, case, image, a.out)
            if dest:
                print(f"   wrote {dest}")
        except Exception as e:  # noqa: BLE001 - report per case, optionally continue
            print(f"   FAILED: {e}", file=sys.stderr)
            failed.append(f"{tool}/{case}")
            if not a.keep_going:
                return 1
    if failed:
        print("failed cases: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
