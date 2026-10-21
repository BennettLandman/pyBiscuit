"""The MaCRUISE-route (and FreeSurfer-route) stage graph.

Only ``ingest`` has a body. Every other stage is registered with its plan rows,
parameters, assets and output names, and raises :class:`StageBlocked` citing the
plan row(s) that must land first. Stage semantics (what exactly each BISCUIT step
computes and in which order) come from the orchestrator spec (brief 11, written from
the MATLAB source Lyu supplied); until it lands the names, inputs and outputs below
are placeholders. Do not fill the semantics in by guessing.
"""

from __future__ import annotations

import json

from .model import Stage, StageContext

__all__ = ["STAGES", "ingest_body"]

_HEMIS = ("lh", "rh")


def _per_hemi(*names: str) -> tuple[str, ...]:
    return tuple(f"{h}.{n}" for h in _HEMIS for n in names)


def ingest_body(ctx: StageContext) -> None:
    """Record the adapter findings (paths, hashes, frames, upstream provenance)."""
    (ctx.attempt_dir / "inputs.json").write_text(
        json.dumps(ctx.findings.to_json(), indent=2, sort_keys=True) + "\n"
    )


STAGES: list[Stage] = [
    Stage(
        name="ingest",
        plan_rows=("B10.3",),
        inputs=("*",),
        params=("surf_recon",),
        outputs=("inputs.json",),
        body=ingest_body,
    ),
    Stage(
        name="surfprep",
        plan_rows=("B5.1", "B5.2", "B5.3", "B5.4", "B5.5", "B5.7"),
        inputs=("ingest:inputs.json", "inner_surface", "seg_refine", "t1"),
        params=("surf_smooth",),
        outputs=_per_hemi("white.surf.gii", "pial.surf.gii", "mid.surf.gii", "sphere.surf.gii",
                          "thickness.func.gii"),
        routes=("macruise",),
        blocked="B5: not specified yet; waits on the orchestrator spec (brief 11, from the "
        "MATLAB source) and the surface-prep port",
    ),
    Stage(
        name="features",
        plan_rows=("B5.6", "B3.2", "B9.1"),
        inputs=("ingest:inputs.json", "surfprep:lh.white.surf.gii"),
        params=("curv_smoothness",),
        outputs=_per_hemi("curv.func.gii", "sulc.func.gii", "iH.func.gii",
                          "inflated.surf.gii"),
        blocked="B5.6: inflation/sulc/iH wait on the orchestrator spec (brief 11) and B3.2 "
        "MeshProperty (needs the Linux image runs, Q1)",
    ),
    Stage(
        name="register",
        plan_rows=("B6.3", "B6.4", "B6.5"),
        inputs=("ingest:inputs.json", "features:lh.curv.func.gii"),
        params=("reg_temp",),
        assets=("reg_template", "icospheres"),
        outputs=_per_hemi("sphere.reg.surf.gii"),
        blocked="B6: HSD is ported (`pybiscuit hsd`) but not wired in: its inputs (B5 spheres "
        "and features) and call sequence wait on the orchestrator spec (brief 11)",
    ),
    Stage(
        name="parc",
        plan_rows=("B7.1", "B7.2", "B7.3", "B7.4", "B7.5"),
        inputs=("ingest:inputs.json", "register:lh.sphere.reg.surf.gii"),
        params=("parc_protocol",),
        assets=("parc_models", "icospheres"),
        outputs=_per_hemi("braincolor.label.gii", "dkt31.label.gii"),
        blocked="B7.1/B7.4/B7.5: feature extraction, graph cut and cleanup are not ported yet "
        "(orchestrator spec, brief 11); B7.2-B7.3 inference exists but is not wired",
    ),
    Stage(
        name="trace",
        plan_rows=("B8.1", "B8.2", "B8.3", "B8.4"),
        inputs=("ingest:inputs.json", "features:lh.curv.func.gii"),
        params=("trace_r", "trace_delta", "curv_smoothness"),
        outputs=_per_hemi("scurve", "gcurve", "scurve.bary", "gcurve.bary"),
        blocked="B8: TRACE is ported (`pybiscuit trace`) but not wired in: its inputs and call "
        "arguments wait on the orchestrator spec (brief 11)",
    ),
    Stage(
        name="quant",
        plan_rows=("B9.1", "B9.2", "B9.4", "B9.5", "B9.6"),
        inputs=("ingest:inputs.json", "features:lh.curv.func.gii", "surfprep:lh.pial.surf.gii"),
        params=("curv_smoothness", "lgi_kernel_size", "lgi_anisotropy", "lgi_ref_area",
                "sci_kernel_size", "sci_bins"),
        outputs=_per_hemi("lgi.func.gii", "sulcal_depth.func.gii", "sci.func.gii",
                          "shape_index.func.gii"),
        blocked="B9: sulcal depth (B9.5) is not ported; LGI (B9.4, brief 07) is not on main "
        "yet; SCI (B9.6) waits on its licence (Q7)",
    ),
    Stage(
        name="stats",
        plan_rows=("B10.1",),
        inputs=("ingest:inputs.json", "parc:lh.dkt31.label.gii", "trace:lh.scurve",
                "quant:lh.lgi.func.gii"),
        outputs=("roi.tsv", "curve.tsv", "cortex.tsv"),
        blocked="B10.1: statistics (surf_quant_stat_pipeline.m) are not ported yet "
        "(orchestrator spec, brief 11)",
    ),
    Stage(
        name="report",
        plan_rows=("B12.3",),
        inputs=("ingest:inputs.json", "stats:roi.tsv"),
        outputs=("biscuit.pdf",),
        blocked="B12.3: report rendering is not implemented (visualization, brief 09)",
    ),
]
