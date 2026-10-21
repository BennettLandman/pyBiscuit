"""Command-line entry point."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pybiscuit")
    p.add_argument("--version", action="version", version=f"pybiscuit {__version__}")
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("info", help="print versions and asset registry status")
    sub.add_parser(
        "spherical-remesh",
        help="port of SphericalRemesh (same flags as the original binary; see its --help)",
        add_help=False,
    )
    sub.add_parser(
        "lgi",
        help="local gyrification index (port of Gyrification; same flags as the original binary)",
        add_help=False,
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "lgi":  # remaining args are passed through verbatim
        from .quant.lgi_cli import main as lgi_main

        return lgi_main(argv[1:])
    if argv and argv[0] == "spherical-remesh":  # remaining args are passed through verbatim
        from .sphere.remesh_cli import main as remesh_main

        return remesh_main(argv[1:])
    args = p.parse_args(argv)
    if args.cmd == "info":
        return _info()
    p.print_help()
    return 0


def _info() -> int:
    import nibabel
    import numba
    import numpy
    import scipy

    print(f"pybiscuit {__version__}")
    print(f"python {sys.version.split()[0]}")
    for m in (numpy, scipy, nibabel, numba):
        print(f"{m.__name__} {m.__version__}")
    try:
        from . import assets

        print(assets.status_summary())
    except ImportError:
        print("assets: registry not available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
