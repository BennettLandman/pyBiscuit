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
    sub.add_parser("register", help="BISCUIT's four-call HSD chain (brief 21)", add_help=False)
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "register":
        from .hsd.chain_cli import main as register_main

        return register_main(argv[1:])
    if argv and argv[0] == "trace":
        from .trace.extract import main as trace_main

        return trace_main(argv[1:])
    if argv and argv[0] == "hsd":  # remaining args are passed through verbatim
        from .hsd.cli import main as hsd_main

        return hsd_main(argv[1:])
    # --- quant (brief 07): lgi, outer-hull; remaining args are passed through verbatim
    for name, text in (
        ("lgi", "local gyrification index (port of Gyrification; flags of the original binary)"),
        ("outer-hull", "cerebral outer hull of a surface (port of OuterHull; provisional)"),
        ("sulcal-depth", "sulcal depth of a pial surface (port of sulcal_depth.m; provisional)"),
        ("lgi-pipeline", "BISCUIT's TRACE + LGI phases on <h>.gray.vtk / <h>.white.vtk (brief 20)"),
        ("sci", "shape complexity index (port of ShapeComplexityIndex; spec §9.4)"),
    ):
        sub.add_parser(name, help=text, add_help=False)
    if argv and argv[0] in ("lgi", "outer-hull", "sulcal-depth", "lgi-pipeline", "sci"):
        if argv[0] == "lgi":
            from .quant.lgi_cli import main as quant_main
        elif argv[0] == "sulcal-depth":
            from .quant.sulcal_depth_cli import main as quant_main
        elif argv[0] == "lgi-pipeline":
            from .quant.lgi_pipeline_cli import main as quant_main
        elif argv[0] == "sci":
            from .quant.sci_cli import main as quant_main
        else:
            from .quant.hull_cli import main as quant_main

        return quant_main(argv[1:])
    # --- end quant
    if argv and argv[0] == "spherical-remesh":  # remaining args are passed through verbatim
        from .sphere.remesh_cli import main as remesh_main

        return remesh_main(argv[1:])
    # --- run (B10.2/B10.3): pipeline driver; options are parsed by pipeline/cli.py ---
    sub.add_parser("run", help="run the pipeline on an input directory", add_help=False)
    if argv and argv[0] == "run":
        from .pipeline.cli import main as run_main

        return run_main(argv[1:])
    # --- end run ---
    # --- viz (brief 09): view/render; implementation lives in pybiscuit.viz.cli ---
    sub.add_parser("view", help="desktop surface viewer (needs pybiscuit[viz])", add_help=False)
    sub.add_parser("render", help="render PNG panels (needs pybiscuit[viz])", add_help=False)
    if argv and argv[0] in ("view", "render"):
        from .viz.cli import main as viz_main

        return viz_main(argv)
    # --- end viz ---
    # --- surfprep (brief 12): mesh-contour, hemi-split; implementation in pybiscuit.surfprep.cli
    sub.add_parser("mesh-contour", help="MeshContour stand-in (cutting plane)", add_help=False)
    sub.add_parser("hemi-split", help="MaCRUISE hemisphere separation", add_help=False)
    if argv and argv[0] in ("mesh-contour", "hemi-split"):
        from .surfprep import cli as surfprep_cli

        if argv[0] == "mesh-contour":
            return surfprep_cli.mesh_contour_main(argv[1:])
        return surfprep_cli.hemi_split_main(argv[1:])
    # --- end surfprep ---
    # --- mesh (brief 13): mesh-property, mesh-deform ---
    sub.add_parser(
        "mesh-property",
        help="port of MeshProperty (same flags as the original binary; see its --help)",
        add_help=False,
    )
    sub.add_parser(
        "mesh-deform",
        help="port of MeshDeform (same flags as the original binary; see its --help)",
        add_help=False,
    )
    if argv and argv[0] == "mesh-property":
        from .mesh.property import main as prop_main

        return prop_main(argv[1:])
    if argv and argv[0] == "mesh-deform":
        from .mesh.deform import main as deform_main

        return deform_main(argv[1:])
    # --- end mesh ---
    # --- parc (brief 15): parc ---
    sub.add_parser(
        "parc",
        help="cortical surface parcellation (U-Net, graph cut, cleanup, fusion)",
        add_help=False,
    )
    if argv and argv[0] == "parc":
        from .parc.cli import main as parc_main

        return parc_main(argv[1:])
    # --- end parc ---
    # --- stats (brief 18): stats ---
    sub.add_parser(
        "stats",
        help="summary ROI and curve statistics (spec §10, B10.1)",
        add_help=False,
    )
    if argv and argv[0] == "stats":
        from .stats.cli import main as stats_main

        return stats_main(argv[1:])
    # --- end stats ---
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
