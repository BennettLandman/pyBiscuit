"""``pybiscuit lgi``: command line of the upstream ``Gyrification`` binary (same flags and outputs).

Writes ``<output>.lgi.map.<kernel>.txt`` (and ``.k1./.k2.`` maps with ``--kArea1/2``,
``--geodesicMap`` file) exactly as the original does.  ``--heap`` is an extra option choosing the
``std::heap`` tie-breaking flavour (see :mod:`pybiscuit.quant._fastmarch`).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .lgi import gyrification, write_geodesic_map, write_lgi_maps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pybiscuit lgi",
        description="Local Gyrification Index (port of Gyrification v1.2.2, Lyu et al. 2018).",
    )
    a = p.add_argument
    a("-i", "--inputSurface", required=True, help="input surface (VTK)")
    a("--outer", required=True, help="outer surface (cerebral hull) with vertex correspondence")
    a("-g", "--gyrus", required=True, help="gyral curves: *.gcurve | *.gcurve.bary")
    a("-s", "--sulcus", required=True, help="sulcal curves: *.scurve | *.scurve.bary")
    a("-c", "--corr", help="correspondence between outer and input (one outer id per vertex)")
    a("--inputMap", help="input geodesic map (skips the distance computation)")
    a("-o", "--output", required=True, help="output file name without extension")
    a("--geodesicMap", help="write the geodesic distance map")
    a("--kArea1", action="store_true", help="write kernel area of the surface per vertex")
    a("--kArea2", action="store_true", help="write kernel area of the hull per vertex")
    a("-t", "--intvArea", type=float, default=0.0, help="write outputs per area interval")
    a("-r", "--radius", type=float, default=30.0, help="search radius (unused upstream)")
    a("-m", "--maxArea", type=float, default=316.0, help="maximum propagation area (rho)")
    a("--speed", type=float, default=0.2, help="speed at sulcal/gyral points (eta)")
    a("--refCortexArea", type=float, default=0.0, help="expected reference cortex area")
    a("--refHullArea", type=float, default=77100.0, help="expected reference hull area")
    a("--nThreads", type=int, default=0, help="number of threads (default OMP_NUM_THREADS or 1)")
    a("--heap", choices=["libcxx", "libstdcxx"], default="libstdcxx", help="std::heap tie order")
    a("--libm", choices=["fdlibm", "host"], default="fdlibm", help="acosf of the normal weights")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for opt in ("maxArea", "speed"):
        if getattr(args, opt) <= 0:
            print(f"error: --{opt} must be positive", file=sys.stderr)
            return 2
    corr = np.loadtxt(args.corr, dtype=np.int64) if args.corr else None
    res = gyrification(
        args.inputSurface,
        args.outer,
        sulcus=args.sulcus,
        gyrus=args.gyrus,
        max_area=args.maxArea,
        intv_area=args.intvArea,
        speed=args.speed,
        ref_cortex_area=args.refCortexArea,
        ref_hull_area=args.refHullArea,
        input_map=args.inputMap,
        corr=corr,
        n_threads=args.nThreads or None,
        heap=args.heap,
        libm=args.libm,
    )
    print(f"Total {res.n_hull_defects} outer hull regions corrected.")
    print(f"  Adjusted ratio: {res.adj_ratio}\n  Adjusted size: {res.max_area}")
    write_lgi_maps(res, args.output)
    if args.geodesicMap:
        write_geodesic_map(args.geodesicMap, res.adj_dist, res.source)
    if args.kArea1:
        write_lgi_maps(res, args.output, kind="k1")
    if args.kArea2:
        write_lgi_maps(res, args.output, kind="k2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
