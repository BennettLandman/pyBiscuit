"""``pybiscuit lgi-pipeline``: BISCUIT's TRACE + LGI phases on ``<h>.gray.vtk`` / ``<h>.white.vtk``.

Mirrors ``curve_extract_pipeline.m`` followed by ``gyrification.m`` / the ``lgi`` script (spec
§8, §9.3), with the ported TRACE in place of the image's ``CurveExtraction`` and the hull
correspondence stand-in in place of ``klaplace`` (see :mod:`pybiscuit.quant.lgi_pipeline`).
Writes ``<out>/curve/<h>.gray.scurve(.bary)``, ``<h>.white.gcurve(.bary)`` and
``<out>/<h>.gray.LGI.txt``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..trace.pipeline import curve_extract_pipeline
from .lgi_pipeline import DEFAULT_KERNEL, DEFAULT_REF_AREA, DEFAULT_SPEED, lgi_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pybiscuit lgi-pipeline",
        description="TRACE curves + local gyrification index as BISCUIT runs them (port).",
    )
    a = p.add_argument
    a("--surf-dir", required=True, help="directory with <h>.gray.vtk (pial) and <h>.white.vtk")
    a("--out", required=True, help="output directory")
    a("--hemi", nargs="+", default=["lh", "rh"], choices=["lh", "rh"])
    a("--trace_r", type=float, default=4.0)
    a("--trace_delta", type=float, default=2.5)
    a("--lgi_kernel_size", type=float, default=DEFAULT_KERNEL)
    a("--lgi_anisotropy", type=float, default=DEFAULT_SPEED)
    a("--lgi_ref_area", type=float, default=DEFAULT_REF_AREA)
    a("--nThreads", type=int, default=1)
    a("--force", action="store_true", help="recompute curves even if they exist")
    args = p.parse_args(argv)
    surf, out = Path(args.surf_dir), Path(args.out)
    curve_dir = out / "curve"
    curve_extract_pipeline(
        surf, curve_dir, hemis=args.hemi, trace_r=args.trace_r, trace_delta=args.trace_delta,
        n_threads=args.nThreads, force=args.force,
    )  # fmt: skip
    summary = {}
    for h in args.hemi:
        run = lgi_pipeline(
            surf / f"{h}.gray.vtk", curve_dir / f"{h}.gray.scurve.bary",
            curve_dir / f"{h}.white.gcurve.bary", out, hemi=h, kernel=args.lgi_kernel_size,
            speed=args.lgi_anisotropy, ref_area=args.lgi_ref_area, n_threads=args.nThreads,
        )  # fmt: skip
        med = float(np.median(run.lgi))
        print(f"{h}: hull area {run.hull_area:.1f} mm^2, LGI median {med:.4f}")
        summary[h] = {"hull_area": run.hull_area, "corr_area": run.corr_area,
                      "lgi_file": run.lgi_file.name}  # fmt: skip
    (out / "lgi_pipeline.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
