"""CLI entrypoint for BISCUIT Phase 4 summary statistics (`B10.1`, spec §10)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_stats_pipeline


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="pybiscuit stats",
        description="BISCUIT Phase 4: ROI/curve summary statistics and registered-space maps.",
    )
    a = p.add_argument
    a(
        "--subject-dir",
        required=True,
        help="Root output directory containing quant/, roi/, curve/, reg/",
    )
    a("--parc-root", required=True, help="Directory containing <protocol>/label.csv")
    a(
        "--uniform-sphere",
        default=None,
        help="Path to sphere_327680.vtk (required unless --no-remesh)",
    )
    a("--stat-dir", default=None, help="Output stat/ directory (defaults to <subject-dir>/stat)")
    a("--protocols", nargs="+", default=["braincolor", "dkt31"], help="Parcellation protocols")
    a(
        "--no-remesh",
        action="store_true",
        help="Skip SphericalRemesh of registered-space property maps",
    )
    a("--force", action="store_true", help="Overwrite existing registered-space property maps")
    args = p.parse_args(argv)

    run_stats_pipeline(
        Path(args.subject_dir),
        stat_dir=Path(args.stat_dir) if args.stat_dir else None,
        parc_root=Path(args.parc_root),
        uniform_sphere=Path(args.uniform_sphere) if args.uniform_sphere else None,
        protocols=tuple(args.protocols),
        remesh_maps=not args.no_remesh,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
