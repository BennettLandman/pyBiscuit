# ruff: noqa: E501
"""``pybiscuit register``: BISCUIT's four-call HSD chain on ``<h>.res.{iH,sulc,H}.txt`` (spec §5.3)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .chain import ChainPaths, chain_argv, run_chain


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pybiscuit register",
        description="BISCUIT surface registration: four chained HSD calls per hemisphere (port).",
    )
    a = p.add_argument
    a("--surf-dir", required=True, help="$J: <h>.res.{iH,sulc,H}.txt in; <h>.coeff.txt, <h>.res.sphere.reg.vtk out")
    a("--template-dir", required=True, help="$ROOT/reg/<template> with <h>.{iH,sulc,curv}.{mean,var}.txt")
    a("--sphere-dir", required=True, help="$ROOT/sphere with sphere_{5120,20480,81920,327680}.vtk")
    a("--hemi", nargs="+", default=["lh", "rh"], choices=["lh", "rh"])
    a("--rh-stage2", choices=["macruise", "lh"], default="macruise",
      help="rh call 2 sphere order: macruise = as the script (swapped), lh = same as lh")  # fmt: skip
    a("--route", choices=["macruise", "fs"], default="macruise",
      help="fs = surf_reg_fs2.m: res.H in call 1, res.curv in call 4, lh sphere order for rh")  # fmt: skip
    a("--nThreads", type=int, default=0)
    a("--robust-aabb", action="store_true", help="opt-in float64 AABB fallback when float32 closestFace misses (Q16)")
    a("--dry-run", action="store_true", help="print the HSD argument lists and exit")
    args = p.parse_args(argv)
    if args.nThreads:
        os.environ["OMP_NUM_THREADS"] = str(args.nThreads)
    for h in args.hemi:
        paths = ChainPaths(h, Path(args.surf_dir), Path(args.template_dir), Path(args.sphere_dir))
        if args.dry_run:
            for i, a_ in enumerate(chain_argv(paths, rh_stage2=args.rh_stage2, route=args.route), 1):
                print(f"[{h} call {i}] HSD " + " ".join(a_))
            continue
        run_chain(
            paths, rh_stage2=args.rh_stage2, route=args.route, n_threads=args.nThreads,
            robust_aabb=args.robust_aabb,
        )  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
