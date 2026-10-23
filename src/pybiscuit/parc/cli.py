"""pybiscuit parc: command-line interface for cortical surface parcellation."""

from __future__ import annotations

import argparse
import sys

from .pipeline import parcellate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pybiscuit parc",
        description="Cortical surface parcellation using spherical U-Net, graph cut, and cleanup.",
    )
    p.add_argument("-i", "--input", required=True, help="subject surface directory")
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="output directory for parcellation label file",
    )
    p.add_argument("--hemi", choices=["lh", "rh"], default="lh", help="hemisphere (lh or rh) [lh]")
    p.add_argument(
        "--protocol",
        choices=["braincolor", "dkt31"],
        default="braincolor",
        help="parcellation protocol [braincolor]",
    )
    p.add_argument("--degree", type=int, default=0, help="alignment spherical harmonics degree [0]")
    p.add_argument("--assets", default=None, help="path to BISCUIT assets directory")
    p.add_argument(
        "--robust-aabb",
        action="store_true",
        help="opt-in float64 SphereAABB fallback (Q16)",
    )
    p.add_argument(
        "--no-quantize",
        action="store_true",
        help="disable intermediate %%f text rounding",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parcellate(
        subject_dir=args.input,
        hemi=args.hemi,
        protocol=args.protocol,
        out_dir=args.output,
        degree=args.degree,
        assets_dir=args.assets,
        robust_aabb=args.robust_aabb,
        quantize=not args.no_quantize,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
