"""``pybiscuit sulcal-depth in.vtk out.txt``: port of ``sulcal_depth.m`` (provisional)."""

from __future__ import annotations

import argparse

import numpy as np

from .sulcal_depth import sulcal_depth, write_depth


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pybiscuit sulcal-depth",
        description="Sulcal depth of a pial surface (port of sulcal_depth.m; provisional).",
    )
    p.add_argument("input", help="pial surface (VTK)")
    p.add_argument("output", help="output per-vertex depth (.txt, one %%f per line)")
    args = p.parse_args(argv)
    sd = sulcal_depth(args.input)
    print(f"Min sulcal depth: {np.nanmin(sd):f}")
    print(f"Max sulcal depth: {np.nanmax(sd):f}")
    write_depth(sd, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
