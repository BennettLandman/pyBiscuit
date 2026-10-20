"""Regenerate tests/data/golden/meshlib/sh_* from the MeshLib SH oracle (needs harness_sh).

Run: .venv/bin/python scripts/oracles/make_golden_sh.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from fixtures.sh_points import sh_fixture_points  # noqa: E402

HARNESS = Path.home() / "dev" / "biscuit" / "oracles" / "meshlib-sh" / "harness_sh"
OUT = ROOT / "tests" / "data" / "golden" / "meshlib"
CASES = {  # name -> (degree, from, which overload stored)
    "sh_deg20_f32": (20, 0, "f32"),
    "sh_deg12_f64": (12, 0, "f64"),
    "sh_deg9_from5_f32": (9, 5, "f32"),
}


def main() -> None:
    pts = sh_fixture_points(n_random=150, ico_level=2)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        pts.tofile(td / "p.bin")
        for name, (deg, frm, which) in CASES.items():
            argv = [str(HARNESS), "p.bin", str(len(pts)), str(deg), str(frm), "o32", "o64"]
            subprocess.run(
                [argv[0], str(td / "p.bin"), *argv[2:5], str(td / "o32"), str(td / "o64")],
                check=True,
            )
            k = (deg + 1) ** 2 - frm * frm
            y = (
                np.fromfile(td / "o32", np.float32)
                if which == "f32"
                else np.fromfile(td / "o64", np.float64)
            ).reshape(len(pts), k)
            d = OUT / name
            d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / "sh.npz", points=pts, y=y)
            cc = subprocess.run(
                ["/usr/bin/clang++", "--version"], capture_output=True, text=True
            ).stdout.splitlines()[0]
            (d / "command.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            "harness_sh",
                            "pts.bin",
                            str(len(pts)),
                            str(deg),
                            str(frm),
                            "out_f32.bin",
                            "out_f64.bin",
                        ],
                        "stored": which,
                        "degree": deg,
                        "from": frm,
                        "commit": "MeshLib dcef807",
                        "compiler": cc + " -O2 -std=c++14",
                        "oracle": "upstream-macos-arm64",
                        "points": "tests/fixtures/sh_points.py sh_fixture_points(n_random=150, ico_level=2)",
                    },
                    indent=2,
                )
                + "\n"
            )
            print(name, y.shape)


if __name__ == "__main__":
    main()
