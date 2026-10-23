"""CLI for Shape Complexity Index (B9.6).

Matches the arguments of ``ShapeComplexityIndex``:
    ``ShapeComplexityIndex -i <mid.vtk> -o <mid.SCI.txt> -k <kernel_size> -s <bins_step>``
"""

from __future__ import annotations

import argparse

from .sci import shape_complexity_index, write_sci


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pybiscuit sci",
        description="Compute Shape Complexity Index (port of ShapeComplexityIndex; spec §9.4).",
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input surface mesh (legacy VTK)",
    )
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output per-vertex SCI scalar map (.txt, %%f per line)",
    )
    p.add_argument(
        "-k",
        "--kernel",
        type=float,
        default=3.0,
        help="Geodesic kernel radius in mm (default: 3.0)",
    )
    p.add_argument(
        "-s",
        "--step",
        type=int,
        default=5,
        help="Step size parameter defining 2^step + 1 bins (default: 5 -> 33 bins)",
    )
    args = p.parse_args(argv)

    sci = shape_complexity_index(args.input, kernel_size=args.kernel, step=args.step)
    write_sci(sci, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
