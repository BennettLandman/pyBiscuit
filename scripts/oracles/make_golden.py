#!/usr/bin/env python
"""Generate committed golden outputs from the upstream oracle binaries.

    .venv/bin/python scripts/oracles/make_golden.py <tool> <case> [--out DIR]
    .venv/bin/python scripts/oracles/make_golden.py --list

Writes ``tests/data/golden/<tool>/<case>/`` containing the (small) oracle outputs
and ``command.json``::

    {"tool", "case", "argv", "commit", "compiler", "oracle": "upstream-macos-arm64",
     "inputs": {...how to regenerate the inputs...}, "outputs": [...files kept...]}

``argv`` is relative to a working directory in which the inputs were generated
(inputs are NOT committed; rebuild them with the registered generator at test
time).  Keep each case well under 1 MB.

Adding a tool (registry contract)
---------------------------------
Define ``def my_tool_case(case: str, work: Path, oracles: Path) -> dict`` that
(1) writes the inputs into ``work``, (2) runs the oracle with cwd=``work`` and
(3) returns ``{"argv": [...], "commit": "<short sha>", "outputs": ["file", ...],
"inputs": {...}}`` where ``outputs`` are paths relative to ``work`` that should
be copied into the golden dir.  Register it with ``@register("my-tool")``.
No other plumbing is needed; ``cases(tool)`` may be added as an attribute
(``my_tool_case.cases = ["a", "b"]``) for ``--list``.
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
GOLDEN = REPO / "tests" / "data" / "golden"
sys.path.insert(0, str(REPO / "src"))

REGISTRY: dict = {}


def register(tool: str):
    def deco(fn):
        REGISTRY[tool] = fn
        return fn

    return deco


def compiler_version() -> str:
    out = subprocess.run(["/usr/bin/clang++", "--version"], capture_output=True, text=True)
    return out.stdout.splitlines()[0] if out.stdout else "unknown"


def run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{argv} failed ({r.returncode}):\n{r.stdout}\n{r.stderr}")
    return r


# ---------------------------------------------------------------- spherical-remesh
SR_CASES = {
    # case: (source level, target level)
    "down5to4": (5, 4),
    "up3to4": (3, 4),  # committed golden (small)
    "up5to6": (5, 6),  # outputs >5 MB: not committed, run live in tests
}
SR_SEED = 20261020


def sr_fixture(work: Path, src_level: int, dst_level: int, seed: int = SR_SEED) -> dict:
    """Write sphere.vtk (icosphere src), ref.vtk (icosphere dst), subject.vtk, prop.txt.

    subject.vtk: the source sphere with a smooth radial bump (radius 1 + 0.2*x*y + 0.1*z)
    so ``-i/-o`` is exercised.  prop.txt: one float per vertex, smooth field
    sin(3x)+cos(2y)+z^2 plus seeded noise (default_rng(seed), sigma 0.01), 6 decimals.
    """
    from pybiscuit.io import write_vtk
    from pybiscuit.mesh import Mesh
    from pybiscuit.sphere.icosphere import icosphere

    src = icosphere(src_level)
    dst = icosphere(dst_level)
    v = src.vertices
    write_vtk(src, work / "sphere.vtk", dtype="float")
    write_vtk(dst, work / "ref.vtk", dtype="float")
    r = 1 + 0.2 * v[:, 0] * v[:, 1] + 0.1 * v[:, 2]
    write_vtk(Mesh(v * r[:, None], src.faces), work / "subject.vtk", dtype="float")
    rng = np.random.default_rng(seed)
    prop = (
        np.sin(3 * v[:, 0])
        + np.cos(2 * v[:, 1])
        + v[:, 2] ** 2
        + 0.01 * rng.standard_normal(len(v))
    )
    (work / "prop.txt").write_text("".join(f"{x:.6f}\n" for x in prop))
    return {
        "generator": "make_golden.sr_fixture",
        "src_level": src_level,
        "dst_level": dst_level,
        "seed": seed,
        "files": ["sphere.vtk", "ref.vtk", "subject.vtk", "prop.txt"],
    }


def sr_argv(oracles: Path) -> list[str]:
    """Command line (relative file names; cwd = fixture dir)."""
    return [
        str(oracles / "spherical-remesh" / "bin" / "SphericalRemesh"),
        "-s", "sphere.vtk", "-i", "subject.vtk", "-r", "ref.vtk", "-p", "prop.txt",
        "-o", "out.vtk", "--outputProperty", "outprop", "--bary", "bary.txt", "--quiet",
    ]  # fmt: skip


@register("spherical-remesh")
def spherical_remesh_case(case: str, work: Path, oracles: Path) -> dict:
    src, dst = SR_CASES[case]
    inputs = sr_fixture(work, src, dst)
    argv = sr_argv(oracles)
    run(argv, work)
    return {
        "argv": argv,
        "commit": "b677c39",
        "outputs": ["out.vtk", "outprop.prop.txt", "bary.txt"],
        "inputs": inputs,
    }


spherical_remesh_case.cases = list(SR_CASES)


# ---------------------------------------------------------------- driver
def make_golden(tool: str, case: str, out_root: Path = GOLDEN, oracles: Path = ORACLES) -> Path:
    if tool not in REGISTRY:
        raise SystemExit(f"unknown tool {tool!r}; registered: {sorted(REGISTRY)}")
    dest = out_root / tool / case
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        res = REGISTRY[tool](case, work, oracles)
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for rel in res["outputs"]:
            shutil.copy(work / rel, dest / Path(rel).name)
        argv = [a.replace(str(oracles), "$ORACLES") for a in res["argv"]]
        meta = {
            "tool": tool,
            "case": case,
            "argv": argv,
            "commit": res["commit"],
            "compiler": compiler_version(),
            "oracle": "upstream-macos-arm64",
            "inputs": res.get("inputs", {}),
            "outputs": [Path(o).name for o in res["outputs"]],
        }
        (dest / "command.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = sum(p.stat().st_size for p in dest.iterdir())
    if size > 1_000_000:
        print(f"WARNING: {dest} is {size} bytes (>1MB)", file=sys.stderr)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("tool", nargs="?")
    ap.add_argument("case", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", type=Path, default=GOLDEN)
    a = ap.parse_args()
    if a.list or not (a.tool and a.case):
        for t, fn in sorted(REGISTRY.items()):
            print(t, getattr(fn, "cases", ["<see docstring>"]))
        return
    print(make_golden(a.tool, a.case, a.out))


if __name__ == "__main__":
    main()
