"""BISCUIT's TRACE phase (``curve_extract_pipeline.m``, spec §8): the ``CurveExtraction`` calls.

``curve_argv`` builds the argument vectors BISCUIT gives ``CurveExtraction`` (``trace_r`` ->
``--epRad`` and ``--nhDist``, ``trace_delta`` -> ``--lsTh``, both through MATLAB `num2str`; every
other option at its default).  :func:`curve_extract_pipeline` runs them through the ported tool
(``pybiscuit.trace.extract.main``), lh then rh, sulcus (pial = ``gray``) then gyrus (white), and
skips a call when both outputs exist (``curve_extract_pipeline.m:37,45``).
"""

from __future__ import annotations

import math
from pathlib import Path

from .extract import main as trace_main

__all__ = ["curve_argv", "curve_extract_pipeline", "num2str"]

DEFAULT_R = 4.0  # biscuit.conf trace_r
DEFAULT_DELTA = 2.5  # biscuit.conf trace_delta


def num2str(x: float) -> str:
    """MATLAB ``num2str`` of a real scalar: integers as ``%d``, otherwise ``%.{p}g`` with
    ``p = max(floor(log10(|x|)) + 5, 5)`` (MATLAB's ``'%11.5g'`` rule for ``|x| < 10``)."""
    if float(x) == int(x):
        return str(int(x))
    digits = max(int(math.floor(math.log10(abs(x)))) + 5, 5)
    return f"{x:.{digits}g}"


def curve_argv(
    kind: str,
    surface,
    prefix,
    *,
    trace_r: float = DEFAULT_R,
    trace_delta: float = DEFAULT_DELTA,
    n_threads: int = 1,
) -> list[str]:
    """Argument vector (without the program name) of one BISCUIT ``CurveExtraction`` call.

    ``kind``: ``"sulcus"`` (pial surface) or ``"gyrus"`` (white surface).
    """
    if kind not in ("sulcus", "gyrus"):
        raise ValueError("kind must be 'sulcus' or 'gyrus'")
    r, d = num2str(trace_r), num2str(trace_delta)
    return [
        f"--{kind}", "-i", str(surface), "-o", str(prefix), "--nThreads", str(n_threads),
        "--epRad", r, "--nhDist", r, "--lsTh", d, "--bary", "--noVtk",
    ]  # fmt: skip


def curve_calls(
    surf_dir, out_dir, *, hemis=("lh", "rh"), trace_r=DEFAULT_R, trace_delta=DEFAULT_DELTA,
    n_threads: int = 1,
) -> list[tuple[list[str], tuple[Path, Path]]]:  # fmt: skip
    """The calls of ``curve_extract_pipeline.m`` in order, with the two files each one produces.

    Inputs: ``<surf_dir>/<h>.gray.vtk`` (sulci) and ``<surf_dir>/<h>.white.vtk`` (gyri); outputs
    ``<out_dir>/<h>.gray.{scurve,scurve.bary}`` and ``<h>.white.{gcurve,gcurve.bary}``.
    """
    surf_dir, out_dir = Path(surf_dir), Path(out_dir)
    calls = []
    for h in hemis:
        for kind, stem, ext in (("sulcus", "gray", "scurve"), ("gyrus", "white", "gcurve")):
            prefix = out_dir / f"{h}.{stem}"
            argv = curve_argv(
                kind, surf_dir / f"{h}.{stem}.vtk", prefix, trace_r=trace_r,
                trace_delta=trace_delta, n_threads=n_threads,
            )  # fmt: skip
            calls.append((argv, (Path(f"{prefix}.{ext}"), Path(f"{prefix}.{ext}.bary"))))
    return calls


def curve_extract_pipeline(
    surf_dir, out_dir, *, hemis=("lh", "rh"), trace_r=DEFAULT_R, trace_delta=DEFAULT_DELTA,
    n_threads: int = 1, force: bool = False,
) -> list[Path]:  # fmt: skip
    """Run BISCUIT's four (or two) TRACE calls with the ported tool; returns the ``.bary`` paths."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    made = []
    for argv, (curve, bary) in curve_calls(
        surf_dir, out_dir, hemis=hemis, trace_r=trace_r, trace_delta=trace_delta,
        n_threads=n_threads,
    ):  # fmt: skip
        if force or not (curve.exists() and bary.exists()):
            if trace_main(argv) != 0:
                raise RuntimeError(f"TRACE failed: pybiscuit trace {' '.join(argv)}")
        made.append(bary)
    return made
