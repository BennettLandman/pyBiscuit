# Agent brief 22 — FreeSurfer-route replay against real image outputs

**Written 2026-10-22.** Read [AGENTS.md](../../AGENTS.md), [plan.md](../../plan.md),
`docs/reference/orchestrator-spec.md` (§5, §8, §9, §12, §13),
[validation-tolerances.md](../validation-tolerances.md) (including the
cross-platform policy, Q14 and Q15), and the [parallel rules](README.md#parallel-sessions).

## Context

The owner added `~/dev/biscuit/open-examples/{0050952,0050961,0050971}`:
ABIDE (NYU) subjects processed with FreeSurfer 7.4.1 and then
`biscuit_FC_v2.2.sif --surf_recon freesurfer` (template `mb101`, protocols
`braincolor,dkt31`, run 2025-09-17). These are the **first real image
outputs** in the project. Each subject has:

- **Inputs:** `surf/{lh,rh}.{white,pial,sphere,inflated}`, `curv`, `sulc`,
  `inflated.H`, `thickness`, `label/?h.cortex.label`.
- **Image outputs:** `surf/?h.{white,gray,mid,inflated,sphere}.vtk`,
  `reg/` (`?h.coeff.txt`, `?h.{sphere,white,gray,mid}.reg.vtk`), `curve/`,
  `quant/` (`gray.LGI`, `gray.SD`, `mid.H`, `mid.S`, `mid.SCI`, `mid.area`,
  `thickness`), `roi/` (`?h.mask`, `?h.parc.{braincolor,dkt31}`), `stat/`
  (`*.reg.*.txt` on the template, `stats.{braincolor,dkt31}.csv`), `pdf/`.
- `tmp/` is empty: no intermediates were kept.

Rules: the data stays outside git (open data, but AGENTS.md forbids subject
data in the repo). Tests that use it are `@pytest.mark.reference` and skip
when `~/dev/biscuit/open-examples` is absent. Commit only metrics JSON and
reports. These outputs are **image evidence**, which ranks above the macOS
oracles in the ground-truth hierarchy.

## Scope

1. **FS-route input stage** (`pybiscuit.surfprep.freesurfer`, spec §12):
   from the FreeSurfer files, produce exactly what
   `run_postproc_fs_pipeline.m` produces. That means the VTKs (`pial` →
   `gray`, the `mid` surface, frames as written by `write_vtk`), the
   `<name>.txt` feature files (`%f\n`), and the mask from `cortex.label`.
   **Accept:** our VTKs match `surf/?h.*.vtk` (vertex max abs ≤ the
   files' print precision, identical faces).
2. **Stage replays.** Drive each merged port with the reproduced inputs and
   compare against the image output, per subject and hemisphere:
   - HSD four-call chain, FS variant (`surf_reg_fs2.m`: `res.H` stage 1,
     `res.curv` stage 4, correct `rh` order) vs `reg/`;
   - TRACE vs `curve/`;
   - LGI (`lgi` chain: OuterHull + klaplace stand-in + Gyrification) vs
     `quant/?h.gray.LGI.txt`;
   - sulcal depth vs `quant/?h.gray.SD.txt`;
   - thickness and area passthroughs vs `quant/`.
   One script, `scripts/validation/open_examples.py`, writes
   `docs/reports/open-examples-YYYY-MM-DD/metrics-<subj>.json` plus a summary
   table.
3. **Judge each stage by the right rule.** Use E-class where the stage is
   stable. Where the original is chaotic (HSD, LGI), measure our own 1-ulp
   input sensitivity on that subject and apply the Q14/Q15 proposals,
   reported as proposed and not adopted. Explain any stage that fails
   before touching code. A real port bug found here gets fixed with a
   regression test. Do not loosen bounds.
4. **Gaps.** For stages still unported (MeshProperty H/S, SCI, graph cut,
   cleanup, stats, PDF), only record which image outputs will serve as their
   references, and which intermediate the replay would need. Do not port
   them here.

## Out of scope

The MaCRUISE route surface prep (B5); porting MeshProperty (brief 13), graph
cut (brief 16), stats (brief 18) or the PDF (brief 19).

## Hand-back

PR to `main` with the FS input stage, the validation script, reference
tests, any port bug fixes, a report summarising every stage across 3
subjects × 2 hemispheres, and updated plan rows: mark a row Done only where
the image comparison passes its stated rule.
