"""``pybiscuit outer-hull``: command line of ``OuterHull`` (``OuterHull in.vtk out.vtk``)."""

from __future__ import annotations

import argparse

from .hull import outer_hull, write_hull


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pybiscuit outer-hull",
        description="Cerebral hull of a cortical surface (port of OuterHull.m; provisional).",
    )
    p.add_argument("input", help="input surface (VTK)")
    p.add_argument("output", help="output hull (VTK)")
    p.add_argument(
        "--grid", type=int, default=256, help="voxels along the longest side (script: 256)"
    )
    args = p.parse_args(argv)
    write_hull(outer_hull(args.input, grid=args.grid), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
