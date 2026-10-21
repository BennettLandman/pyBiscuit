# Agent briefs

One brief = one working session. Each brief lists its scope, acceptance,
evidence, and stop conditions; AGENTS.md rules always apply.

| Brief | Scope | Depends on | Status |
|---|---|---|---|
| [01](01-foundations.md) | Scaffold, I/O, topology, icospheres, MeshConv operators | — | Done (PR #1, merged) |
| [02](02-geometry-remesh.md) | Oracles, MeshLib geometry, AABB, SH basis, SphericalRemesh | 01 | Done (PR #2, merged) |
| [03](03-portable-numerics.md) | Cross-platform comparison policy; get CI green | 02 | Done (in PR #2, merged) |
| [04](04-geodesics-trace.md) | Geodesics + TRACE | 03 | Done (PR #4, merged) |
| [05](05-hsd.md) | HSD registration | 03 | Done (PR #5, merged) |
| [06](06-parcellation-net.md) | Parcellation network inference | 03 | Done (PR #3, merged) |
| [07](07-lgi.md) | OuterHull + Gyrification | 03 | Done (PR #6, merged) |
| [08](08-pipeline-infra.md) | Config, stage graph, resume, run.json, input adapters | 03 | Done (PR #8, merged) |
| [09](09-viz-foundation.md) | GIfTI exports, renderer, NiiVue HTML viewer | 03 | Done (PR #9, merged) |
| [10](10-reference-harness.md) | Image reference scripts + Actions workflow (not run) | 03 | Done (PR #7, merged; not run, needs Q1) |
| [11](11-orchestrator-spec.md) | Orchestrator spec from MATLAB source; reconcile 04–07 | — | Done (PR #10, merged) |
| [12](proposed-12-21.md#12--maccruise-hemisphere-separation-b52) | MaCRUISE hemisphere separation | 11 | Done (PR #12, merged; provisional) |
| [13](proposed-12-21.md#13--meshproperty--meshdeform-black-box-port-and-feature-maps-b31b33-b91) | MeshProperty/MeshDeform black-box port, feature maps | 11, image goldens (runbook job 1) | Proposed |
| [14](proposed-12-21.md#14--freesurfer-replacement-surface-prep-b53b57) | FreeSurfer-replacement surface prep | 12, 13 (Q3 answered: close enough) | Proposed |
| [15](proposed-12-21.md#15--parcellation-features-alignment-and-fusion-b71-b73) | Parcellation features, alignment, fusion | 13 | Proposed |
| [16](proposed-12-21.md#16--graph-cut-and-roi-cleanup-b74-b75) | Graph cut + ROI cleanup | 06 (Q13 answered: independent α-expansion) | Proposed (card) |
| [17](proposed-12-21.md#17--sulcal-depth-b95) | Sulcal depth | 07 | Done (PR #11, merged; provisional) |
| [18](proposed-12-21.md#18--statistics-legacy-output-view-pipeline-wiring-b101-b104-integration) | Statistics, legacy view, pipeline wiring | 08, 12–17 | Proposed |
| [19](proposed-12-21.md#19--pdf--qa-layout-b123) | PDF/QA layout | 09, 18 | Proposed |
| [20](proposed-12-21.md#20--lgi-on-the-ported-trace-follow-up-to-07) | LGI on the ported TRACE | 04, 07 | Running |
| [21](proposed-12-21.md#21--hsd-follow-up-biscuits-four-call-chain-follow-up-to-05) | HSD four-call replay | 05 | Queued |

Unblocked by the MATLAB source (2026-10-21); brief 11 wrote the spec ([outlines of the port briefs](proposed-12-21.md)): B5 (surface prep), B7.1/B7.4/B7.5
(parc features, graph cut, cleanup), B9.5 (sulcal depth), B10 (stats,
pipeline). Blocked on Linux image runs (Q1): B1, B3 (MeshProperty).

## Parallel sessions

Briefs 04–07 may run at the same time, each in its own git worktree and
branch, all from `main` (PRs #1–#2 merged 2026-10-20, `e6c223a`).

- **Own files only.** Each brief creates its own modules, tests, golden
  folders, port notes, and report. Shared files need care:
  - `plan.md`: edit only your own rows' Status cells. Do not edit the top
    Status section; the owner or a coordinating session updates it after
    merges.
  - `src/pybiscuit/cli.py`: add your subcommand in one self-contained block.
    Expect a trivial merge conflict.
  - `pyproject.toml`: new dependencies need owner sign-off (AGENTS.md); add
    extras only in your own section.
  - `tests/_compare.py` and `src/pybiscuit/mesh/*` from earlier briefs:
    read-only. If you need a change, put it in your report as a request.
- **Oracle rebuilds** go into `~/dev/biscuit/oracles/<tool>-nofma/`, one
  directory per tool, so sessions don't overwrite each other's binaries.
- Merge order is whatever finishes first. Rebase onto `main` before opening
  the PR.

## Launch queue (owner instruction 2026-10-20)

When a running session finishes, the coordinating session verifies its PR
(tests, CI, report) and launches the next queued brief on **Sonnet**, in its
own worktree from `main`. The coordinating session works only in `~/dev/biscuit/pyBiscuit-coord`, never in another session's checkout:

1. 08 pipeline infrastructure → 2. 09 visualization → 3. 10 reference harness.
4. **11** orchestrator spec (MATLAB source arrived 2026-10-21), then the port briefs it proposes.
5. Follow-ups: LGI on the ported TRACE (after 07 merges), and HSD part 2 if brief 05's report needs one. Number them after brief 11's proposals.

Merges: the owner approved merging verified PRs (full six-job CI matrix + local suite on the merge result).

Concurrency: at most **two** brief sessions at once, since three exhausted the shared usage limit on 2026-10-21. Queue after 12/17: 20 → 16 → 21. Image-dependent briefs (13, then 14/15) start when runbook job 1 results come back.
