# ruff: noqa: E501
"""``pybiscuit hsd``: same flags as the upstream ``HSD`` binary (wrapper/PARSE_ARGS.h @ f169595).

The image binary exposes exactly these flags (checked with ``strings``): the landmark/location
options and the ``*Dir`` / ``--filter`` batch forms are commented out upstream.  The landmark and
location options and their ``--landmarkDir`` / ``--surfaceDir`` forms are offered here as opt-in
extras (``UPSTREAM_DISABLED``), clearly not part of the original binary; without them the run is
the image binary's.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HSD_VERSION = "v1.3.9"
#: upstream options commented out in wrapper/PARSE_ARGS.h (absent from the image binary), exposed opt-in
UPSTREAM_DISABLED = ("-l", "--landmark", "--surface", "--weightMap", "--locationWeight", "--landmarkDir", "--surfaceDir")
DESC = (
    f"Hierarchical Spherical Deformation for Cortical Surface Registration {HSD_VERSION}\n"
    "Author: Ilwoo Lyu (port: pybiscuit)\n"
)


def _nonneg_int(s):
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def _nonneg_float(s):
    v = float(s)
    if v < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return v


def _pos_int(s):
    v = int(s)
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


def _ico(s):
    v = int(s)
    if not 0 <= v <= 7:
        raise argparse.ArgumentTypeError("must be in [0, 7]")
    return v


def _file(s):
    if not Path(s).is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {s}")
    return s


def _dir(s):
    if not Path(s).is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {s}")
    return s


def _list_dir(path: str, suffix: str) -> list[str]:
    """``getListFile`` (main.cpp): entries ending in ``suffix``, ``path + "/" + name``, sorted."""
    return sorted(path + "/" + f for f in os.listdir(path) if f.endswith(suffix))


def _trim(lst: list[str], names: list[str]) -> list[str]:
    """``getTrimmedList`` (main.cpp): keep entries containing one of the subject names."""
    keys = [n[n.rfind("/") + 1 :] for n in names]
    return sorted(e for e in lst if any(k in e for k in keys))


def _subject_names(spheres: list[str]) -> list[str]:
    """main.cpp: file name without directory and the last 4 characters, cut at the first '.'."""
    out = []
    for p in spheres:
        pivot = p.rfind("/") + 1
        name = p[pivot : len(p) - 4]
        dot = name.find(".")
        out.append(name if dot == -1 else name[:dot])
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pybiscuit hsd", description=DESC)
    p.add_argument("-s", "--sphere", nargs="+", required=True, type=_file, help="list of sphere files")
    p.add_argument("-p", "--property", nargs="+", required=True, type=_file, help="list of property files (subject-major)")
    p.add_argument("-o", "--output", nargs="+", default=[], help="list of output sphere files")
    p.add_argument("--outputcoeff", nargs="+", default=[], help="list of output coeff files")
    p.add_argument("-c", "--coefficient", nargs="+", default=[], type=_file, help="previous SH coefficient files")
    p.add_argument("-d", "--degree", type=_nonneg_int, default=15, help="SH degree [15]")
    p.add_argument("--icosahedron", type=_ico, default=7, help="icosahedral subdivision level of the sampling points [7]")
    p.add_argument("--weight", nargs="+", type=_nonneg_float, default=[], help="weight of each property")
    p.add_argument("--idprior", type=_nonneg_float, default=625.0, help="inverse of the distortion prior [625]")
    p.add_argument("--maxIter", type=_nonneg_int, default=20, help="max iterations at the final phase [20]")
    p.add_argument("--icomesh", type=_file, default=None, help="pre-defined icosahedral mesh (overrides --icosahedron)")
    p.add_argument("--gres", type=_pos_int, default=3, help="search resolution for the initial guess [3]")
    p.add_argument(
        "--guess-raw-icomesh", action="store_true",
        help="upstream behaviour: the guess queries the raw (radius-100) --icomesh mesh, so every query misses",
    )  # fmt: skip
    p.add_argument(
        "--robust-aabb", action="store_true",
        help="opt-in float64 AABB fallback when float32 closestFace misses near an edge (Q16)",
    )  # fmt: skip
    p.add_argument("--noguess", action="store_true", help="skip the rigid initial guess")
    p.add_argument("--resample", action="store_true", help="resample properties on the current icosahedral level")
    p.add_argument("--writecoeff", action="store_true", help="write coefficients whenever the energy decreases")
    p.add_argument("--fixedSubjects", nargs="+", type=_nonneg_int, default=[], help="indices of subjects not deformed")
    p.add_argument("--tmpVar", type=_file, default=None, help="prior of feature variance (pair-wise only)")
    p.add_argument("--nThreads", type=_nonneg_int, default=0, help="threads (0: OMP_NUM_THREADS or all cores)")
    x = p.add_argument_group(
        "upstream-disabled options (NOT in the original HSD binary)",
        "Commented out in upstream wrapper/PARSE_ARGS.h and absent from the BISCUIT image; ported for "
        "completeness, validated against an oracle built with them enabled (docs/ports/hsd.md).",
    )
    x.add_argument("-l", "--landmark", nargs="+", default=[], type=_file, help="list of landmark files (vertex indices), one per subject")
    x.add_argument("--landmarkDir", type=_dir, default=None, help="directory of landmark files (*txt, matched to the sphere names)")
    x.add_argument("--surface", nargs="+", default=[], type=_file, help="list of surface model files for location information")
    x.add_argument("--surfaceDir", type=_dir, default=None, help="directory of surface model files (*vtk, sorted)")
    x.add_argument("--weightMap", type=_nonneg_float, default=1.0, help="overall property weight (eta), used with landmarks [1]")
    x.add_argument("--locationWeight", type=float, default=0.0, help="weighting factor of location information [0]")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.surfaceDir and not a.surface:
        a.surface = _list_dir(a.surfaceDir, "vtk")
    if a.landmarkDir and not a.landmark:
        a.landmark = _trim(_list_dir(a.landmarkDir, "txt"), _subject_names(a.sphere))
    if not a.surface:
        a.locationWeight = 0.0  # main.cpp: if (surf == NULL) weightLoc = 0;
    nsubj = len(a.sphere)
    nprop_total = len(a.property)
    n_out, n_oc, nfixed = len(a.output), len(a.outputcoeff), len(a.fixedSubjects)
    weight = a.weight or [1.0] * (nprop_total // nsubj)
    fail = None
    if nsubj != n_oc and nsubj != n_out:
        fail = "Fatal error: # of subjects is incosistent with # of outputs!"
    elif nprop_total // nsubj != len(weight):
        fail = "Fatal error: # of properties is incosistent with # of weighting factors!"
    elif nsubj == nfixed:
        fail = "Fatal error: no further optimization - all subjects are fixed!"
    elif nprop_total % nsubj:
        fail = "Fatal error: # of properties is not a multiple of # of subjects!"
    elif any(i >= nsubj for i in a.fixedSubjects):
        fail = f"Fatal error: wrong index of the fixed subject! {max(a.fixedSubjects)}"
    elif a.coefficient and len(a.coefficient) != nsubj:
        fail = "Fatal error: # of coefficient files must equal # of subjects"
    elif (n_out and n_out != nsubj) or (n_oc and n_oc != nsubj):
        fail = "Fatal error: # of outputs must equal # of subjects"
    elif a.landmark and len(a.landmark) != nsubj:
        fail = "Fatal error: # of landmark files must equal # of subjects"
    elif a.locationWeight > 0 and len(a.surface) != nsubj:
        fail = "Fatal error: # of surface files must equal # of subjects"
    if fail:
        print(fail)
        return 1
    nprop = nprop_total // nsubj
    props = [a.property[s * nprop : (s + 1) * nprop] for s in range(nsubj)]

    import numba

    nthr = a.nThreads or int(os.environ.get("OMP_NUM_THREADS", 0)) or numba.config.NUMBA_NUM_THREADS
    numba.set_num_threads(max(1, min(nthr, numba.config.NUMBA_NUM_THREADS)))

    from .register import HSD

    hsd = HSD(
        a.sphere, props, output=a.output or None, output_coeff=a.outputcoeff or None, weight=weight,
        degree=a.degree, idprior=a.idprior, coefficient=a.coefficient or None, max_iter=a.maxIter,
        fixed_subjects=a.fixedSubjects or None, icosahedron_level=a.icosahedron,
        realtime_coeff=a.writecoeff, tmp_variance=a.tmpVar, guess=not a.noguess,
        guess_res=a.gres, ico_mesh=a.icomesh, resampling=a.resample,
        guess_unit_ico=not a.guess_raw_icomesh, robust_aabb=a.robust_aabb,
        landmarks=a.landmark or None, surfaces=a.surface if a.locationWeight > 0 else None,
        weight_loc=a.locationWeight, weight_map=a.weightMap,
    )  # fmt: skip
    hsd.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
