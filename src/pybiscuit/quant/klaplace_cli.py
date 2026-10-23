"""CLI for klaplace surface correspondence (B9.3).

Matches the arguments of ``klaplace``:
    ``klaplace [-thread <N>] [-dims 128] <source> <target> -surfaceCorrespondence <prefix>``
    ``klaplace -conv <input> <output>``
"""

from __future__ import annotations

import argparse
import sys

from .klaplace import klaplace, klaplace_conv


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pybiscuit klaplace",
        description=(
            "Surface correspondence via Laplace PDE streamlines (port of klaplace; spec §9.3)."
        ),
    )
    p.add_argument(
        "-conv",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="Convert mesh file (e.g. from vtp to vtk)",
    )
    p.add_argument(
        "-thread",
        "--thread",
        type=int,
        default=1,
        help="Number of threads (ignored; numba parallelizes automatically)",
    )
    p.add_argument(
        "-dims",
        "--dims",
        type=int,
        default=128,
        help="Grid resolution (default: 128)",
    )
    p.add_argument(
        "-surfaceCorrespondence",
        "--surfaceCorrespondence",
        dest="prefix",
        help="Output prefix for surface correspondence",
    )
    p.add_argument(
        "positional",
        nargs="*",
        help="Positional arguments [source target] or [source target prefix]",
    )
    p.add_argument("-s", "--source", help="Source (pial) surface mesh")
    p.add_argument("-t", "--target", help="Target (hull) surface mesh")
    p.add_argument("-p", "--prefix", help="Output prefix")
    p.add_argument("--iters", type=int, default=10000, help="Jacobi iterations (default: 10000)")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    if args.conv:
        klaplace_conv(args.conv[0], args.conv[1])
        return 0

    source = args.source
    target = args.target
    prefix = args.prefix

    pos = list(args.positional)
    if not source and pos:
        source = pos.pop(0)
    if not target and pos:
        target = pos.pop(0)
    if not prefix and pos:
        prefix = pos.pop(0)

    if not source or not target or not prefix:
        raise ValueError(
            "klaplace requires source mesh, target mesh, and -surfaceCorrespondence <prefix>."
        )

    klaplace(source, target, prefix, dims=args.dims, n_iters=args.iters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
