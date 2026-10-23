# ruff: noqa: E501
"""BISCUIT's four chained ``HSD`` calls (``surf_reg_macruise.m:55-66,109-120``; spec §5.3) and the
parcellation alignment call (``parc.m:61``; spec §6.3), built as argv for :mod:`pybiscuit.hsd.cli`.

One property per call (iH, sulc, sulc, curv(=-4H)), template first and fixed (``--fixedSubjects 0``), degree 10,
``--icomesh`` levels 4 -> 7 selected by sphere file, the coefficient file ``C`` chained through ``-c C C
--outputcoeff C C`` (a fixed subject neither reads nor writes it), ``--tmpVar`` only in the two finest calls.
Call 4 also writes the registered sphere with ``-o R R``.

``rh`` call 2 is ``-s S81920 S327680`` in the MaCRUISE and CIVET scripts (``:112``) against ``-s S327680 S81920`` for
``lh`` (and for ``surf_reg_fs2.m``); ``rh_stage2="macruise"`` reproduces it, ``"lh"`` uses the ``lh`` order.

``route="fs"`` is ``surf_reg_fs2.m`` (spec §12): call 1 takes ``<h>.res.H.txt`` (the resampled FreeSurfer
``inflated.H``) in place of ``res.iH``, call 4 takes ``<h>.res.curv.txt`` (the resampled FreeSurfer ``curv``) in
place of ``res.H``, and ``rh`` call 2 has the ``lh`` sphere order.  The calls are otherwise the same.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath


def _p(x) -> str:
    """argv path with forward slashes (valid on Windows too; matches the MATLAB-built strings of the spec)."""
    return PurePath(x).as_posix()


#: sphere files by name in ``$ROOT/sphere`` -> vertices / icosahedral level
SPHERES = {"sphere_5120": 2562, "sphere_20480": 10242, "sphere_81920": 40962, "sphere_327680": 163842}

#: (subject sphere, --icomesh sphere, feature, template stem, --tmpVar) of the four calls (lh order)
_SCHEDULE = (
    ("sphere_20480", "sphere_5120", "iH", "iH", False),
    ("sphere_81920", "sphere_20480", "sulc", "sulc", False),
    ("sphere_327680", "sphere_81920", "sulc", "sulc", True),
    ("sphere_327680", "sphere_327680", "H", "curv", True),
)


@dataclass(frozen=True)
class ChainPaths:
    """File layout of one hemisphere's registration (names as in the MaCRUISE script)."""

    hemi: str
    surf_dir: Path  # ``$J``: ``<h>.res.{iH,sulc,H}.txt`` in, ``<h>.coeff.txt`` / ``<h>.res.sphere.reg.vtk`` out
    template_dir: Path  # ``$ROOT/reg/<temp>``: ``<h>.{iH,sulc,curv}.{mean,var}.txt``
    sphere_dir: Path  # ``$ROOT/sphere``: ``sphere_{5120,20480,81920,327680}.vtk``

    @property
    def coeff(self) -> Path:
        return self.surf_dir / f"{self.hemi}.coeff.txt"

    @property
    def registered(self) -> Path:
        return self.surf_dir / f"{self.hemi}.res.sphere.reg.vtk"

    def sphere(self, name: str) -> Path:
        return self.sphere_dir / f"{name}.vtk"

    def template(self, stem: str, kind: str) -> Path:
        return self.template_dir / f"{self.hemi}.{stem}.{kind}.txt"

    def feature(self, name: str) -> Path:
        return self.surf_dir / f"{self.hemi}.res.{name}.txt"


#: feature file names (``<h>.res.<name>.txt``) of calls 1 and 4 on the FreeSurfer route
_FS_FEATURES = {0: "H", 3: "curv"}


def chain_argv(
    paths: ChainPaths, *, rh_stage2: str = "macruise", route: str = "macruise"
) -> list[list[str]]:
    """The four ``HSD`` argument lists (without the program name), as strings."""
    if rh_stage2 not in ("macruise", "lh"):
        raise ValueError("rh_stage2 must be 'macruise' or 'lh'")
    if route not in ("macruise", "fs"):
        raise ValueError("route must be 'macruise' or 'fs'")
    if route == "fs":
        rh_stage2 = "lh"
    c = _p(paths.coeff)
    out = []
    for i, (subj, ico, feat, stem, var) in enumerate(_SCHEDULE):
        spheres = ["sphere_327680", subj]
        if i == 1 and paths.hemi == "rh" and rh_stage2 == "macruise":
            spheres = [subj, "sphere_327680"]  # template/subject spheres swapped (surf_reg_macruise.m:112)
        a: list[str] = []
        if i:
            a += ["-c", c, c]
        a += ["-s", *(_p(paths.sphere(s)) for s in spheres)]
        if i == 3:
            r = _p(paths.registered)
            a += ["-o", r, r]
        a += ["--outputcoeff", c, c]
        if route == "fs":
            feat = _FS_FEATURES.get(i, feat)
        a += ["-p", _p(paths.template(stem, "mean")), _p(paths.feature(feat))]
        a += ["--fixedSubjects", "0", "-d", "10", "--icomesh", _p(paths.sphere(ico))]
        if var:
            a += ["--tmpVar", _p(paths.template(stem, "var"))]
        out.append(a)
    return out


def parc_alignment_argv(
    sphere_327680: str | Path, native_sphere: str | Path, curv_mean: str | Path, h_txt: str | Path
) -> list[str]:
    """``parc.m:61``: rigid refinement of the native sphere to the parcellation protocol's curvature (``-o`` = input)."""
    n = _p(native_sphere)
    return [
        "-s", _p(sphere_327680), n, "-o", n, n, "-p", _p(curv_mean), _p(h_txt),
        "--fixedSubjects", "0", "--noguess", "-d", "0", "--icosahedron", "5",
    ]  # fmt: skip


def run_chain(
    paths: ChainPaths,
    *,
    rh_stage2: str = "macruise",
    route: str = "macruise",
    n_threads: int = 0,
    stages: Sequence[int] = (1, 2, 3, 4),
    runner=None,
    guess_raw_icomesh: bool = False,
    robust_aabb: bool = False,
) -> list[list[str]]:
    """Run the chain with the ported ``HSD`` (or ``runner(argv) -> int``); returns the argv lists run.

    ``guess_raw_icomesh`` adds the port-only flag that reproduces upstream master's call-1 initial guess (the
    macOS oracle's behaviour), which the image does not show (brief 22, Q17); see ``docs/ports/hsd.md`` H-3.
    ``robust_aabb`` adds ``--robust-aabb`` (opt-in float64 AABB fallback, Q16).
    """
    if runner is None:
        from .cli import main as runner  # noqa: N813
    argvs = chain_argv(paths, rh_stage2=rh_stage2, route=route)
    done = []
    for i in stages:
        a = argvs[i - 1] + (["--nThreads", str(n_threads)] if n_threads else [])
        if guess_raw_icomesh:
            a = a + ["--guess-raw-icomesh"]
        if robust_aabb:
            a = a + ["--robust-aabb"]
        rc = runner(a)
        if rc:
            raise RuntimeError(f"HSD call {i} for {paths.hemi} failed (exit {rc})")
        done.append(a)
    return done
