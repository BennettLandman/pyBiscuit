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
    sub.add_parser("trace", help="port of CurveExtraction/TRACE (original flags)", add_help=False)
    sub.add_parser(
        "hsd",
        help="port of HSD spherical registration (flags as the original binary)",
        add_help=False,
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "trace":
        from .trace.extract import main as trace_main

        return trace_main(argv[1:])
    if argv and argv[0] == "hsd":  # remaining args are passed through verbatim
        from .hsd.cli import main as hsd_main

        return hsd_main(argv[1:])
    if argv and argv[0] == "spherical-remesh":  # remaining args are passed through verbatim
        from .sphere.remesh_cli import main as remesh_main

        return remesh_main(argv[1:])
    # --- viz (brief 09): view/render; implementation lives in pybiscuit.viz.cli ---
    sub.add_parser("view", help="desktop surface viewer (needs pybiscuit[viz])", add_help=False)
    sub.add_parser("render", help="render PNG panels (needs pybiscuit[viz])", add_help=False)
    if argv and argv[0] in ("view", "render"):
        from .viz.cli import main as viz_main

        return viz_main(argv)
    # --- end viz ---
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
