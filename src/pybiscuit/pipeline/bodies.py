"""Stage bodies of the BISCUIT pipeline (plan B10.2).

Each body reproduces one phase of ``biscuit.m`` (orchestrator spec §2) inside its stage
attempt directory, under BISCUIT's own file names (``<h>.white.vtk``, ``<h>.res.H.txt``,
``<h>.coeff.txt``, ``<h>.parc.dkt31.txt``, ``<h>.gray.scurve.bary``, ``<h>.mid.H.txt`` …), so the
``legacy/`` view (B10.4) can be assembled from the attempt directories and every
intermediate can be compared with the image's. Upstream files are copied in by name.

The bodies call the validated per-tool ports; they never write placeholder results. A
missing input or a failing tool raises, and the runner records the stage as failed.

- ``surfprep_body``: MaCRUISE surface prep (spec §4; B5.1-B5.7)
- ``features_body``: job-directory surfaces and registration features (spec §5.1, §12)
- ``register_body``: four-call HSD chain and re-tessellation (spec §5.3-5.5)
- ``parc_body``: parcellation per protocol (spec §6)
- ``trace_body``: TRACE curves (spec §8)
- ``quant_body``: H, S, area, sulcal depth, LGI, SCI, thickness (spec §9)
- ``stats_body``: ROI/curve statistics and registered-space maps (spec §10)
- ``report_body``: QA report (spec §11)
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pybiscuit.io.gifti import write_func_gii, write_surf_gii
from pybiscuit.io.vtk import read_vtk
from pybiscuit.pipeline.model import StageContext

__all__ = [
    "features_body",
    "parc_body",
    "quant_body",
    "register_body",
    "report_body",
    "stats_body",
    "surfprep_body",
    "trace_body",
]

_HEMIS = ("lh", "rh")
#: surfaces of the job directory (``pial`` is ``gray`` in BISCUIT's names)
_SURFACES = ("white", "gray", "mid", "inflated", "sphere")


class StageInputError(RuntimeError):
    """A file a stage needs from upstream is missing."""


def _find_stage_attempt_dir(ctx: StageContext, stage_name: str) -> Path | None:
    """Return the latest attempt directory of ``stage_name``."""
    stage_root = ctx.attempt_dir.parents[1] / stage_name
    if not stage_root.is_dir():
        return None
    attempts = sorted(d for d in stage_root.glob("attempt-*") if d.is_dir())
    return attempts[-1] if attempts else None


def _find_stage_output(ctx: StageContext, stage_name: str, filename: str) -> Path | None:
    att = _find_stage_attempt_dir(ctx, stage_name)
    if att is not None and (att / filename).is_file():
        return att / filename
    return None


def _pull(ctx: StageContext, stage_name: str, filenames: Sequence[str]) -> None:
    """Copy ``filenames`` from the latest attempt of ``stage_name`` into this attempt."""
    att = _find_stage_attempt_dir(ctx, stage_name)
    missing = []
    for name in filenames:
        src = att / name if att is not None else None
        if src is None or not src.is_file():
            missing.append(name)
            continue
        dst = ctx.attempt_dir / name
        if not dst.exists():
            shutil.copyfile(src, dst)
    if missing:
        raise StageInputError(f"{ctx.stage.name}: missing from stage {stage_name!r}: {missing}")


def _find_assets_root() -> Path:
    """Locate the BISCUIT assets directory containing ``reg/``, ``parc/`` and ``sphere/``."""
    candidates = []
    if env := os.environ.get("PYBISCUIT_ASSETS"):
        candidates.append(Path(env))
    from pybiscuit.assets import cache_dir

    candidates.append(cache_dir())
    repo_root = Path(__file__).resolve().parents[3]
    dev_root = repo_root.parent
    candidates.extend(
        [
            repo_root / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit",
            dev_root / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit",
            repo_root / "reference" / "converted",
            dev_root / "reference" / "converted",
        ]
    )
    for c in candidates:
        if c.is_dir() and (c / "sphere").is_dir() and (c / "reg").is_dir():
            return c
    raise StageInputError(
        "BISCUIT assets (reg/, parc/, sphere/) not found; set PYBISCUIT_ASSETS or run "
        "`pybiscuit fetch-assets`"
    )


def _uniform_sphere() -> Path:
    return _find_assets_root() / "sphere" / "sphere_327680.vtk"


def _threads(ctx: StageContext) -> int:
    return int(ctx.config.get("omp_num_threads", 1) or 1)


def _route(ctx: StageContext) -> str:
    return "fs" if ctx.findings.surf_recon == "freesurfer" else "macruise"


def _protocols(ctx: StageContext) -> list[str]:
    protocols = ctx.params.get("parc_protocol", ["braincolor", "dkt31"])
    if isinstance(protocols, str):
        protocols = [p.strip() for p in protocols.split(",") if p.strip()]
    return list(protocols)


def _txt(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


def _resample_feature(sphere: Path, feature: Path, out_prefix: Path) -> None:
    """T18/T20: ``SphericalRemesh -s <sphere> -r sphere_327680 -p <feature> --outputProperty``."""
    from pybiscuit.sphere.remesh import run as remesh_run

    remesh_run(sphere, ref=_uniform_sphere(), property=[feature], output_property=out_prefix,
               quiet=True)  # fmt: skip


# ------------------------------------------------------------------ phase 1a (MaCRUISE)
def surfprep_body(ctx: StageContext) -> None:
    """MaCRUISE route: hemisphere separation, volumes, white/pial/mid/inflated/sphere (spec §4)."""
    from pybiscuit.io.vtk import read_vtk as _read
    from pybiscuit.surfprep import run_macruise_surfprep

    paths = {k: ctx.inputs.get(k) for k in ("inner_surface", "seg_refine", "t1")}
    missing = [k for k, p in paths.items() if p is None or not Path(p).exists()]
    if missing:
        raise StageInputError(f"surfprep: missing inputs {missing}")
    run_macruise_surfprep(
        inner_surface=paths["inner_surface"],
        seg_refine=paths["seg_refine"],
        t1=paths["t1"],
        out_dir=ctx.attempt_dir,
        surf_smooth=int(ctx.params.get("surf_smooth", 0)),
    )
    for h in _HEMIS:
        for s in ("white", "pial", "mid", "sphere"):
            mesh = _read(ctx.attempt_dir / f"{h}.{'gray' if s == 'pial' else s}.vtk")
            write_surf_gii(ctx.attempt_dir / f"{h}.{s}.surf.gii", mesh, hemi=h, kind=s)
        thick = _txt(ctx.attempt_dir / f"{h}.thickness.txt")
        write_func_gii(ctx.attempt_dir / f"{h}.thickness.func.gii", thick, hemi=h,
                       names=["thickness"])  # fmt: skip


# ------------------------------------------------------------------ phase 1b: job dir + features
def features_body(ctx: StageContext) -> None:
    """The job directory of ``surf_reg_*.m``: surfaces, masks, thickness, registration features.

    FreeSurfer route (spec §12, ``run_postproc_fs_pipeline.m``): surfaces and curv files from
    the ``recon-all`` directory; features ``res.H`` (from ``inflated.H``), ``res.sulc``,
    ``res.curv``. MaCRUISE route (spec §5.1): ``H`` = −4·MeshProperty of white, ``iH`` from the
    inflated surface, ``sulc`` from surfprep; features ``res.{iH,sulc,H}``.
    """
    from pybiscuit.mesh.property import (
        reg_inflated_mean_curvature,
        reg_white_mean_curvature,
        write_property_txt,
    )

    adir = ctx.attempt_dir
    route = _route(ctx)
    if route == "fs":
        from pybiscuit.surfprep.freesurfer import prepare_fs_subject

        prepare_fs_subject(Path(ctx.findings.root), adir, hemis=_HEMIS)
        res_sources = ("curv", "inflated.H", "sulc")
    else:
        for h in _HEMIS:
            _pull(ctx, "surfprep", [f"{h}.{s}.vtk" for s in _SURFACES]
                  + [f"{h}.sulc.txt", f"{h}.thickness.txt", f"{h}.mask.txt"])  # fmt: skip
            white = read_vtk(adir / f"{h}.white.vtk")
            write_property_txt(reg_white_mean_curvature(white, round_intermediate=True),
                               adir / f"{h}.H.txt")  # fmt: skip
            iH = reg_inflated_mean_curvature(read_vtk(adir / f"{h}.inflated.vtk"))
            write_property_txt(iH, adir / f"{h}.iH.txt")
            # parc.m reads <h>.inflated.H (spec §6.2); on the MaCRUISE route that is FreeSurfer's
            # inflated-surface curvature, which the port approximates by iH (open: B5.6).
            shutil.copyfile(adir / f"{h}.iH.txt", adir / f"{h}.inflated.H.txt")
        res_sources = ("iH", "sulc", "H")

    for h in _HEMIS:
        for name in res_sources:
            _resample_feature(adir / f"{h}.sphere.vtk", adir / f"{h}.{name}.txt", adir / f"{h}.res")
        curv = _txt(adir / f"{h}.{'curv' if route == 'fs' else 'H'}.txt")
        ih = _txt(adir / f"{h}.{'inflated.H' if route == 'fs' else 'iH'}.txt")
        write_func_gii(adir / f"{h}.curv.func.gii", curv, hemi=h, names=["curv"])
        write_func_gii(adir / f"{h}.sulc.func.gii", _txt(adir / f"{h}.sulc.txt"), hemi=h,
                       names=["sulc"])  # fmt: skip
        write_func_gii(adir / f"{h}.iH.func.gii", ih, hemi=h, names=["iH"])
        write_surf_gii(adir / f"{h}.inflated.surf.gii", read_vtk(adir / f"{h}.inflated.vtk"),
                       hemi=h, kind="inflated")  # fmt: skip


# ------------------------------------------------------------------ registration
def register_body(ctx: StageContext) -> None:
    """Four HSD calls (T21), native sphere deformation (T22), re-tessellation (T23), areas (T25)."""
    from pybiscuit.hsd.chain import ChainPaths, run_chain
    from pybiscuit.mesh.geometry import vertex_areas
    from pybiscuit.mesh.property import write_property_txt
    from pybiscuit.sphere.remesh import run as remesh_run

    adir = ctx.attempt_dir
    route = _route(ctx)
    root = _find_assets_root()
    template_dir = root / "reg" / str(ctx.params.get("reg_temp", "mb101"))
    if not template_dir.is_dir():
        raise StageInputError(f"registration template not found: {template_dir}")
    uni = _uniform_sphere()
    feats = ("H", "sulc", "curv") if route == "fs" else ("iH", "sulc", "H")

    for h in _HEMIS:
        _pull(ctx, "features", [f"{h}.{s}.vtk" for s in ("white", "gray", "mid", "sphere")]
              + [f"{h}.res.{f}.txt" for f in feats])  # fmt: skip
        paths = ChainPaths(hemi=h, surf_dir=adir, template_dir=template_dir,
                           sphere_dir=root / "sphere")  # fmt: skip
        run_chain(paths, route=route, n_threads=_threads(ctx))
        sphere = adir / f"{h}.sphere.vtk"
        remesh_run(sphere, coeff=paths.coeff, deform=adir / f"{h}.sphere.reg.vtk", deg=10,
                   quiet=True)  # fmt: skip
        for s in ("gray", "white", "mid"):
            remesh_run(sphere, input=adir / f"{h}.{s}.vtk", coeff=paths.coeff, ref=uni,
                       output=adir / f"{h}.{s}.reg.vtk", deg=10, quiet=True)  # fmt: skip
        mid_reg = read_vtk(adir / f"{h}.mid.reg.vtk")
        write_property_txt(vertex_areas(mid_reg.vertices.astype(np.float32), mid_reg.faces),
                           adir / f"{h}.mid.reg.area.txt")  # fmt: skip
        write_surf_gii(adir / f"{h}.sphere.reg.surf.gii", read_vtk(adir / f"{h}.sphere.reg.vtk"),
                       hemi=h, kind="sphere.reg")  # fmt: skip


# ------------------------------------------------------------------ parcellation
def parc_body(ctx: StageContext) -> None:
    """``parc.m`` per protocol (spec §6); the cortical mask goes to ``roi/`` (spec §6.9)."""
    from pybiscuit.parc.pipeline import parcellate

    adir = ctx.attempt_dir
    root = _find_assets_root()
    protocols = _protocols(ctx)
    for h in _HEMIS:
        _pull(ctx, "features", [f"{h}.white.vtk", f"{h}.sphere.vtk", f"{h}.inflated.H.txt",
                                f"{h}.sulc.txt"])  # fmt: skip
        _pull(ctx, "register", [f"{h}.coeff.txt"])
        labels = {}
        for proto in protocols:
            labels[proto] = parcellate(
                subject_dir=adir, hemi=h, protocol=proto, out_dir=adir,
                assets_dir=root, sphere_template=_uniform_sphere(),
            )  # fmt: skip
            write_func_gii(adir / f"{h}.{proto}.label.gii", np.asarray(labels[proto], np.int32),
                           hemi=h, names=[proto])  # fmt: skip
        mask = _find_stage_output(ctx, "features", f"{h}.mask.txt")
        if mask is not None:
            shutil.copyfile(mask, adir / f"{h}.mask.txt")
        else:
            # FreeSurfer route without label/<h>.cortex.label: mask = parc > 0
            # (run_postproc_fs_pipeline.m:113-119); the source does not say which protocol.
            m = (np.asarray(labels[protocols[0]]) > 0).astype(int)
            (adir / f"{h}.mask.txt").write_text("".join(f"{x}\n" for x in m))


# ------------------------------------------------------------------ TRACE
def trace_body(ctx: StageContext) -> None:
    """``CurveExtraction`` on gray (sulci) and white (gyri) (spec §8)."""
    from pybiscuit.trace.pipeline import curve_extract_pipeline

    adir = ctx.attempt_dir
    _pull(ctx, "features", [f"{h}.{s}.vtk" for h in _HEMIS for s in ("gray", "white")])
    curve_extract_pipeline(
        surf_dir=adir,
        out_dir=adir,
        hemis=_HEMIS,
        trace_r=ctx.params.get("trace_r", 4),
        trace_delta=ctx.params.get("trace_delta", 2.5),
        n_threads=_threads(ctx),
    )


# ------------------------------------------------------------------ quantification
def quant_body(ctx: StageContext) -> None:
    """``surf_quant_pipeline.m`` (spec §9): mid H/S/area, gray SD, gray LGI, mid SCI, thickness."""
    from pybiscuit.mesh.property import quant_mid_features, write_property_txt
    from pybiscuit.quant.lgi_pipeline import lgi_pipeline
    from pybiscuit.quant.sci import shape_complexity_index, write_sci
    from pybiscuit.quant.sulcal_depth import sulcal_depth, write_depth

    adir = ctx.attempt_dir
    for h in _HEMIS:
        _pull(ctx, "features", [f"{h}.mid.vtk", f"{h}.gray.vtk", f"{h}.thickness.txt"])
        _pull(ctx, "trace", [f"{h}.gray.scurve.bary", f"{h}.white.gcurve.bary"])
        mid, gray = adir / f"{h}.mid.vtk", adir / f"{h}.gray.vtk"

        n_smooth = int(ctx.params.get("curv_smoothness", 2))
        H, S, area = quant_mid_features(read_vtk(mid), curv_smoothness=n_smooth)
        write_property_txt(H, adir / f"{h}.mid.H.txt")
        write_property_txt(S, adir / f"{h}.mid.S.txt")
        write_property_txt(area, adir / f"{h}.mid.area.txt")

        sd = sulcal_depth(gray)
        write_depth(sd, adir / f"{h}.gray.SD.txt")

        lgi = lgi_pipeline(
            gray,
            adir / f"{h}.gray.scurve.bary",
            adir / f"{h}.white.gcurve.bary",
            adir,
            hemi=h,
            kernel=float(ctx.params.get("lgi_kernel_size", 316)),
            speed=float(ctx.params.get("lgi_anisotropy", 0.2)),
            ref_area=float(ctx.params.get("lgi_ref_area", 77100)),
            n_threads=_threads(ctx),
        ).lgi

        sci = shape_complexity_index(mid, kernel_size=float(ctx.params.get("sci_kernel_size", 3)),
                                     step=int(ctx.params.get("sci_bins", 5)))  # fmt: skip
        write_sci(sci, adir / f"{h}.mid.SCI.txt")

        write_func_gii(adir / f"{h}.lgi.func.gii", lgi, hemi=h, names=["lgi"])
        write_func_gii(adir / f"{h}.sulcal_depth.func.gii", sd, hemi=h, names=["sulcal_depth"])
        write_func_gii(adir / f"{h}.sci.func.gii", sci, hemi=h, names=["sci"])
        write_func_gii(adir / f"{h}.shape_index.func.gii", S, hemi=h, names=["shape_index"])


# ------------------------------------------------------------------ statistics
_MEASURES = ("MeanCurvature", "ShapeIndex", "SulcalDepth", "Thickness", "SCI", "LGI")


def stats_body(ctx: StageContext) -> None:
    """``surf_quant_stat_pipeline.m`` per protocol + registered-space maps (spec §10)."""
    from pybiscuit.stats.pipeline import run_stats_pipeline

    adir = ctx.attempt_dir
    protocols = _protocols(ctx)
    dirs = {s: _find_stage_attempt_dir(ctx, s) for s in ("quant", "parc", "trace", "register")}
    if missing := [s for s, d in dirs.items() if d is None]:
        raise StageInputError(f"stats: no completed attempt of {missing}")
    tbls = run_stats_pipeline(
        adir,
        quant_dir=dirs["quant"],
        roi_dir=dirs["parc"],
        curve_dir=dirs["trace"],
        reg_dir=dirs["register"],
        stat_dir=adir,
        parc_root=_find_assets_root() / "parc",
        uniform_sphere=_uniform_sphere(),
        protocols=protocols,
        remesh_maps=True,
    )
    _pull(ctx, "register", [f"{h}.mid.reg.area.txt" for h in _HEMIS])

    t = tbls[protocols[0]]
    with open(adir / "roi.tsv", "w") as fh:
        fh.write("ROI\t" + "\t".join(_MEASURES) + "\n")
        for i in range(len(t.ids)):
            fh.write(f"{t.labels[i]}\t" + "\t".join(f"{t.avg[i, m]:.6f}" for m in range(6)) + "\n")
    with open(adir / "curve.tsv", "w") as fh:
        fh.write("ROI\tnVertices_curve\tMeanCurvature_curve\tThickness_curve\n")
        for i in range(len(t.ids)):
            fh.write(f"{t.labels[i]}\t{t.n_vertices_curve[i]}\t"
                     f"{t.avg_curve[i, 0]:.6f}\t{t.avg_curve[i, 3]:.6f}\n")  # fmt: skip
    with open(adir / "cortex.tsv", "w") as fh:
        fh.write("Measure\tWholeCortex\tLeftHemi\tRightHemi\n")
        row = {int(t.ids[i]): i for i in range(len(t.ids))}
        for m, name in enumerate(_MEASURES):
            vals = [f"{t.avg[row[k], m]:.6f}" if k in row else "NaN" for k in (20000, 20001, 20002)]
            fh.write(f"{name}\t" + "\t".join(vals) + "\n")


# ------------------------------------------------------------------ report
def report_body(ctx: StageContext) -> None:
    """QA report PDFs merged into ``biscuit.pdf`` (spec §11)."""
    from pybiscuit.viz.report import generate_report_pdfs

    out_root = ctx.attempt_dir.parents[2]
    generate_report_pdfs(
        out_root,
        template=str(ctx.params.get("reg_temp", ctx.config.get("reg_temp", "mb101"))),
        protocols=_protocols(ctx),
        project=str(ctx.config.get("project") or "Project"),
        subject=str(ctx.config.get("subject") or "Subject"),
        session=str(ctx.config.get("session") or "Session"),
    )
    shutil.copyfile(out_root / "PDF" / "biscuit.pdf", ctx.attempt_dir / "biscuit.pdf")
