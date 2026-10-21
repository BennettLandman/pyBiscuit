"""``pybiscuit hsd``: same flags as the upstream ``HSD`` binary (wrapper/PARSE_ARGS.h @ f169595).

The image binary exposes exactly these flags (checked with ``strings``): the landmark/location
options and the ``*Dir`` / ``--filter`` batch forms are commented out upstream.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HSD_VERSION = "v1.3.9"
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
    p.add_argument("--icomesh", type=_file, default="", help="pre-defined icosahedral mesh (overrides --icosahedron)")
    p.add_argument("--gres", type=_pos_int, default=3, help="search resolution for the initial guess [3]")
    p.add_argument("--noguess", action="store_true", help="skip the rigid initial guess")
    p.add_argument("--resample", action="store_true", help="resample properties on the current icosahedral level")
    p.add_argument("--writecoeff", action="store_true", help="write coefficients whenever the energy decreases")
    p.add_argument("--fixedSubjects", nargs="+", type=_nonneg_int, default=[], help="indices of subjects not deformed")
    p.add_argument("--tmpVar", type=_file, default="", help="prior of feature variance (pair-wise only)")
    p.add_argument("--nThreads", type=_nonneg_int, default=0, help="threads (0: OMP_NUM_THREADS or all cores)")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
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
        realtime_coeff=a.writecoeff, tmp_variance=a.tmpVar or None, guess=not a.noguess,
        guess_res=a.gres, ico_mesh=a.icomesh or None, resampling=a.resample,
    )  # fmt: skip
    hsd.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
