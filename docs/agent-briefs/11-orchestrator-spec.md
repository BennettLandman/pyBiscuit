# Agent brief 11 — Orchestrator spec from the BISCUIT MATLAB source (B1.3, B5.1)

**Written 2026-10-21.** Docs and analysis only: no ports. Read
[AGENTS.md](../../AGENTS.md), [plan.md](../../plan.md),
[source-map.md](../source-map.md), [io-contract.md](../io-contract.md),
and the [parallel rules](README.md#parallel-sessions).

## Context

Ilwoo Lyu supplied the BISCUIT source. It is at `~/dev/biscuit/BISCUIT-master/`
(outside git; never commit it, and quote at most short excerpts in docs). Facts
established on 2026-10-21:

- It covers every encrypted `.m` in [ctf-manifest.tsv](../reference/ctf-manifest.tsv).
  The entry point exists in two variants: `BISCUIT/FreeSurfer/biscuit.m` and
  `BISCUIT/GenericRecon/biscuit.m`.
- **`GenericRecon` matches the FC image.** Its `check_input` is identical, and
  its `run_biscuit` and `biscuit.conf` differ only in the final ghostscript PDF
  step and in subfolder input paths (`Surfaces_FreeView/…`, `Surfaces_MNI/…`,
  `Raw/T1.nii.gz`, `SegRefine/…` vs the image's flat names). The parc `test.sh`,
  `model.py` and `meshcnn/ops.py` are byte-identical to the FC CTF.
- Files are dated 2022-02-10, eight months after the image build (2021-06-25).
  Drift is possible. **The image remains the acceptance reference;** the
  source is the spec.

## Deliverables

1. `docs/reference/orchestrator-spec.md`: for the **MaCRUISE route** first,
   then FreeSurfer and CIVET briefly. Cover:
   - Stage order from `biscuit.m` → `run_postproc_macruise_pipeline.m` →
     `surf_reg_macruise.m` → `parc.m` → `curve_extract_pipeline.m` →
     `surf_quant_pipeline.m` → `surf_quant_stat_pipeline.m` → PDF.
   - **Every external tool call**: exact argv as built (HSD, CurveExtraction,
     Gyrification/`lgi`, klaplace, OuterHull, MeshProperty, MeshDeform,
     MeshContour, ShapeComplexityIndex, SphericalRemesh, FreeSurfer `mri_*`/`mris_*`,
     python `test.sh`), each with its `biscuit.conf` parameter mapping
     (`trace_r`, `trace_delta`, `curv_smoothness`, `lgi_*`, `sci_*`,
     `surf_smooth`, `reg_temp`, `parc_protocol`).
   - All files read and written per stage (names, formats, frames), and the
     complete **output tree** (this replaces B1.3's trace-based plan;
     the image run later confirms it).
   - In-MATLAB computation the port must reproduce: hemisphere separation,
     the use of `mc_surf_mni` / `aff.txt`, features (sulc/curv/iH
     definitions), parcellation feature extraction and normalisation, graph cut,
     ROI correction, label fusion, sulcal depth, VOXELISE, statistics, and the
     visualisation layouts.
2. **Reconcile with merged and in-flight ports.** One table per tool comparing
   what BISCUIT actually passes against what the port supports or assumed.
   Cover TRACE (brief 04), HSD's template mean/var invocation (05),
   parcellation inputs and fusion (06), and LGI parameters and klaplace (07).
   Flag every mismatch as an issue for that component; do not edit their code.
3. **Answer or narrow open questions** Q3 (which FreeSurfer steps run and why),
   Q4 (what the MNI surface is for), and Q9 (legacy output names). Put findings
   in open-questions.md's "Why it matters" column as dated facts, and leave
   the Answer column for the owner.
4. **Updated plan rows:** rewrite B5.x, B7.1, B7.4–B7.5, B9.5, B10.x and B12.3
   acceptance text where the source changes what must be built, with each
   change cited to a source file:line. Mark B5.1 and B1.3 (source-derived)
   Done with evidence. B1.3 still needs image confirmation; note that.
5. **Proposed next briefs:** short outlines (scope, dependencies, acceptance)
   for the port work this spec unlocks. Expected: MaCRUISE surface prep (B5),
   features, parcellation features + graph cut + cleanup (B7.1/B7.4/B7.5),
   sulcal depth (B9.5), statistics (B10.1), pipeline wiring onto brief 08's
   machinery, and the PDF layout (B12.3). Add them to the brief index as
   "Proposed".

## Rules

- Licence: BISCUIT is MIT (© 2019 Ilwoo Lyu). Excerpts must keep attribution.
  Don't copy whole files into the repo.
- MATLAB toolbox calls (e.g. `imfill`, `bwconncomp`, `knnsearch`, graph
  functions) must be listed with their exact semantics and options, because
  the Python equivalents differ in defaults (connectivity, tie-breaking).
- Record any place the source disagrees with image behaviour already
  measured in this repo (e.g. brief 01's `.asc` frame findings).

## Hand-back

Branch from `main`; docs-only PR. Report `docs/reports/orchestrator-spec-YYYY-MM-DD.md`
(coverage checklist of all 83 source `.m` files: read, summarized,
irrelevant). Final message: the top surprises, the mismatch table for briefs
04–07, and the proposed next briefs.
