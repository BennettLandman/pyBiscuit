"""``SphericalRemesh`` command line (flag-for-flag port of ``wrapper/PARSE_ARGS.h``).

Flags (identical names/semantics to upstream, b677c39)::

    --quiet
    -s,--sphere (required)  -i,--input  -c,--coeff  -r,--ref  -p,--property (list)
    --color
    -o,--output  --outputProperty  --deform  --bary  --ftype
    --nneighbor  --deg  --deg0  --keepColor

Exit codes follow CLI11: 0 ok, 105 validation error (missing file, bad value),
106 missing required option, 109 unknown argument, 114 argument mismatch.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import sys
from pathlib import Path

VERSION = "v1.2.1"  # SREMESH_VERSION of the pinned upstream build
_DESC = f"Spherical Remeshing Tool {VERSION}\nAuthor: Ilwoo Lyu\n"


class _ArgError(Exception):
    def __init__(self, msg: str, code: int) -> None:
        super().__init__(msg)
        self.code = code


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # noqa: D401
        code = 109 if "unrecognized" in message else 114
        raise _ArgError(message, code)


def _nonneg(s: str) -> int:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{s!r} is not a number") from None
    if v < 0:
        raise argparse.ArgumentTypeError(f"Number {s} not in range")
    return int(v)


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="SphericalRemesh",
        description=_DESC,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--quiet", action="store_true", help="Do not print outputs")
    g = p.add_argument_group("Inputs")
    g.add_argument(
        "-s",
        "--sphere",
        help="Specify spherical model (unit sphere) used for the original spherical parameterization (vtk)",
    )
    g.add_argument("-i", "--input", help="Specify input subject surface model (vtk)")
    g.add_argument("-c", "--coeff", help="Specify HSD spherical harmonic coefficients (txt)")
    g.add_argument(
        "-r",
        "--ref",
        help="Specify reference model (unit sphere) for the final surface remeshing (vtk)",
    )
    g.add_argument(
        "-p",
        "--property",
        nargs="+",
        action="extend",
        default=[],
        help="Specify properties that are included in the resulting mesh files (txt)",
    )
    g.add_argument(
        "--color",
        help="Specify color map of the reference model that will assign the same color to the generated surface (txt)",
    )
    g = p.add_argument_group("Outputs")
    g.add_argument("-o", "--output", help="Specify output surface model (vtk)")
    g.add_argument(
        "--outputProperty",
        help="Specify output of properties after remeshing (no extension needed)",
    )
    g.add_argument("--deform", help="Specify deformed unit sphere by HSD deformation (vtk)")
    g.add_argument("--bary", help="Specify barycentric coordinates (txt)")
    g.add_argument(
        "--ftype", help="Specify data type - output will be saved as a binary format (dat)"
    )
    g = p.add_argument_group("Remeshing parameters")
    g.add_argument("--nneighbor", action="store_true", help="Enable nearest neighbor interpolation")
    g.add_argument(
        "--deg", type=_nonneg, default=-1, help="Specify level of HSD spherical harmonics"
    )
    g.add_argument(
        "--deg0",
        type=_nonneg,
        default=-1,
        help="Specify base level of HSD spherical harmonics to begin with",
    )
    g.add_argument("--keepColor", action="store_true", help="Keep color of the template model")
    return p  # fmt: skip


def parse(argv: list[str]) -> argparse.Namespace:
    """Parse and validate like CLI11 (required ``-s``, ExistingFile checks, ``--ftype`` set)."""
    from .remesh import check_ftype

    a = build_parser().parse_args(argv)
    if a.sphere is None:
        raise _ArgError("--sphere is required", 106)
    for flag in ("sphere", "input", "coeff", "ref", "color"):
        v = getattr(a, flag)
        if v is not None and not Path(v).is_file():
            raise _ArgError(f"--{flag}: File does not exist: {v}", 105)
    for v in a.property:
        if not Path(v).is_file():
            raise _ArgError(f"--property: File does not exist: {v}", 105)
    if a.ftype is not None and a.ftype != "":
        try:
            check_ftype(a.ftype)
        except ValueError as e:
            raise _ArgError(f"--ftype: {e}", 105) from None
    return a


def main(argv: list[str] | None = None) -> int:
    """Run SphericalRemesh; returns the process exit code."""
    from .remesh import run

    argv = list(sys.argv[1:] if argv is None else argv)
    if any(x in ("-h", "--help") for x in argv):
        print(build_parser().format_help())
        return 0
    try:
        a = parse(argv)
    except _ArgError as e:
        print(f"{e}\nRun with --help for more information.", file=sys.stderr)
        return e.code
    run(
        a.sphere,
        input=a.input,
        coeff=a.coeff,
        ref=a.ref,
        property=a.property,
        color=a.color,
        output=a.output,
        output_property=a.outputProperty,
        deform=a.deform,
        bary=a.bary,
        ftype=a.ftype,
        nneighbor=a.nneighbor,
        deg=a.deg,
        deg0=a.deg0,
        keep_color=a.keepColor,
        quiet=a.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
