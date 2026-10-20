# Agent briefs

One brief = one working session. Each brief lists its scope, acceptance,
evidence, and stop conditions; AGENTS.md rules always apply.

| Brief | Scope | Depends on | Status |
|---|---|---|---|
| [01](01-foundations.md) | Scaffold, I/O, topology, icospheres, MeshConv operators | — | Done (PR #1) |
| [02](02-geometry-remesh.md) | Oracles, MeshLib geometry, AABB, SH basis, SphericalRemesh | 01 | Done (PR #2, CI red) |
| [03](03-portable-numerics.md) | Cross-platform comparison policy; get CI green | 02 | Next (sequential) |
| [04](04-geodesics-trace.md) | Geodesics + TRACE | 03 | Ready |
| [05](05-hsd.md) | HSD registration | 03 | Ready |
| [06](06-parcellation-net.md) | Parcellation network inference | 03 | Ready |
| [07](07-lgi.md) | OuterHull + Gyrification | 03 | Ready |

Blocked until the MATLAB source arrives: B5 (surface prep), B7.1/B7.4/B7.5
(parc features, graph cut, cleanup), B9.5 (sulcal depth), B10 (stats,
pipeline). Blocked on Linux image runs (Q1): B1, B3 (MeshProperty).

## Parallel sessions

Briefs 04–07 may run at the same time, each in its own git worktree and
branch, all from `main` after PRs #1–#2 (with brief 03) are merged.

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
