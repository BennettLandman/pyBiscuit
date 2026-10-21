# Agent guide — pyBiscuit

Read this file first, then [plan.md](plan.md). This file holds stable working
rules; plan.md holds status and the step table. Do not duplicate status here.

## Mission

Port BISCUIT (MaCRUISE route first) and every tool it depends on to a single
Python package, `pybiscuit`, installable with `pip`, with no MATLAB, MCR,
FreeSurfer, Singularity, TeX, or compiled C++ build required. Input comes from
PyMaCRUISE. Output reproduces BISCUIT's scientific products within the
tolerances in [docs/validation-tolerances.md](docs/validation-tolerances.md),
plus visualization ([docs/visualization-plan.md](docs/visualization-plan.md)).

## Local layout (owner workstation)

```
~/dev/biscuit/
├── pyBiscuit/                  ← this repository (BennettLandman/pyBiscuit, private)
├── biscuit_FC_v2.2.sif         ← production image (BISCUIT 2.2 = CBT v1.3 + templates + models)
├── biscuit_cbt_v1.3.sif        ← base image (no reg/ or parc/ assets)
├── BISCUIT-master/             ← BISCUIT MATLAB source from Ilwoo Lyu (NOT in git)
└── reference/                  ← NOT in git; regenerate with scripts/extract_reference.py
    ├── sif_FC_v2.2/root/opt/   ← extracted /opt/biscuit and /opt/cmorph
    ├── ctf_plain/              ← plaintext files from the MATLAB CTF (parc Python, meshes)
    ├── repos/                  ← upstream clones (ilwoolyu/*, NIRALUser/ShapeComplexityIndex)
    └── vuiis_pdfs/             ← example BISCUIT QA PDFs
~/dev/mipav/pymacruise/          ← PyMaCRUISE (input producer); read its AGENTS/plan conventions
~/dev/mipav/sample-macruise/     ← sample data incl. legacy MaCRUISE v3 outputs (BISCUIT-ready)
```

Never commit SIF images, extracted binaries, model weights, templates, or
subject data. `reference/` and large assets stay outside git; record hashes.

## Ground truth hierarchy

1. **Behavior of `biscuit_FC_v2.2.sif`** on matched inputs (run on Linux x86-64
   with Apptainer; see [docs/reference-artifacts.md](docs/reference-artifacts.md)).
   This is the acceptance reference.
2. **Upstream C++/Python source** (MIT) in `reference/repos/` and
   `reference/ctf_plain/`. Note that upstream `master` may differ from the
   binaries in the image (CBT v1.3 / cmorph v1.7, June 2021). When they
   disagree, the image binary wins; record the discrepancy.
3. **Papers** (listed in [docs/source-map.md](docs/source-map.md)) explain intent.
4. **BISCUIT MATLAB source** (from Ilwoo Lyu, 2026-10-21) is at
   `~/dev/biscuit/BISCUIT-master/` (outside git; don't commit it).
   `BISCUIT/GenericRecon` is the variant the FC image runs. It is the
   **specification** for orchestration and in-MATLAB computation, but the
   files are dated 2022-02, after the image (2021-06), so image behavior still
   decides acceptance. Never decrypt the CTF;
   [docs/reference/ctf-manifest.tsv](docs/reference/ctf-manifest.tsv) maps it.

## Working rules

- Execute plan rows in dependency order. A row is **Done** only when its
  acceptance check passes and evidence (test name, metrics, report path,
  commit) is recorded in the row. Never pre-fill validation dates.
- Port classes (see source-map): **exact** components (HSD, TRACE, Gyrification,
  klaplace, OuterHull, SCI, MeshProperty, SphericalRemesh, parcellation net)
  must match the reference binary on identical inputs within numeric tolerance.
  **Replacement** components (FreeSurfer `mri_*`/`mris_*` and anything we
  re-design) are judged on downstream-measure agreement. Do not silently move a
  component between classes; that is an owner decision.
- Preserve source behavior, defaults, and stopping rules. Do not "improve"
  algorithms during the port. Put proposed improvements in
  `docs/proposals/` with evidence; they need owner approval.
- Clean-room rule for FreeSurfer: do not read or translate FreeSurfer source
  code. Implement replacements from papers and black-box behavior only
  (FreeSurfer's license is not MIT-compatible).
- Dependencies: NumPy, SciPy, Numba, NiBabel are core. PyTorch only in the
  `[parc]` extra. PyVista/matplotlib only in the `[viz]` extra. Any new runtime
  dependency needs a one-line justification in the PR and a license check
  (MIT/BSD/Apache/PSF only; no GPL/LGPL in the default install).
- Python 3.10–3.13, pure Python + Numba. No C/C++ extensions in the default
  install. Optional accelerated backends must have a pure-Python fallback that
  produces the same results.
- Determinism: fixed seeds, stable sort orders, documented float precision.
  Record every place the original uses float32 vs float64; match it when it
  affects tie-breaking (Dijkstra, argmax, thresholds).
- Coordinates: all surfaces in RAS+ millimetres (same as PyMaCRUISE). Record
  any conversion from legacy frames (FreeSurfer tkRAS, MNI) in the manifest.
- Tests: synthetic fixtures (spheres, ellipsoids, analytic folded surfaces,
  icospheres) committed with generator + seed; small golden outputs from the
  reference binaries committed with provenance. Real-subject data never in git.
- Each module gets a port note in `docs/ports/<component>.md`: source files
  and line ranges ported, deviations, numeric-precision notes, validation link.
- Reports go in `docs/reports/<topic>-YYYY-MM-DD.md`; plan rows link to them.
- Keep plan.md's status section current at the end of each working session.

## Conventions

- Package: `src/pybiscuit/`; CLI entry point `pybiscuit` (subcommands
  `run`, `stage`, `report`, `view`, `fetch-assets`, and one subcommand per
  ported tool mirroring the original binary's flags, e.g. `pybiscuit hsd`,
  `pybiscuit trace`, `pybiscuit lgi`).
- Mesh data model: `Mesh(vertices: float64[N,3], faces: int32[M,3])` plus
  named per-vertex arrays. One canonical I/O module for VTK legacy (ASCII and
  binary), FreeSurfer surf/curv/asc, GIfTI, and PyMaCRUISE VTK headers.
- Each stage writes into `<out>/stages/<name>/` with a `provenance.json`;
  the run writes `run.json` (schema shared in spirit with PyMaCRUISE's).
- Commit messages: imperative, reference the plan row ID (e.g. `B4.2: port HSD
  spherical harmonics basis`).

## When blocked

Record the blocker in plan.md (row status **Blocked** + reason) and add it to
[docs/open-questions.md](docs/open-questions.md). Do not substitute a generic
algorithm for an exact-class component to get unblocked.
