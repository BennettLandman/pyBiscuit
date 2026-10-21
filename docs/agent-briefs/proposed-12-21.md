# Proposed briefs 12–21 (from the orchestrator spec)

**Proposed 2026-10-21 by brief 11.** Outlines only: each one needs a full brief (context, security notes, stop conditions) before it
is launched, written by the coordinating session. They follow the [parallel rules](README.md#parallel-sessions): own files only, one worktree per brief,
Sonnet unless a design decision needs more. Source for every statement: [orchestrator-spec](../reference/orchestrator-spec.md) (cited as *spec §n*) and
[the brief 11 report](../reports/orchestrator-spec-2026-10-21.md) (*report §3.x* for the mismatch ids).
Image acceptance (Q1) is still pending for every E-class row, so rows finish as "oracle-provisional" or "provisional (no oracle)" until then.

| Brief | Scope | Plan rows | Depends on | Can start |
|---|---|---|---|---|
| 12 | MaCRUISE hemisphere separation | B5.2 | 11 | now |
| 13 | MeshProperty / MeshDeform black-box port and feature maps | B3.1–B3.3, B9.1, parts of B5.6 | 11, Q1 probes for semantics | now (oracle-free parts), full acceptance after Q1 |
| 14 | FreeSurfer-replacement surface prep | B5.3–B5.7 | 12, 13, owner answer to Q3 | after Q3 |
| 15 | Parcellation features, alignment and DKT31 fusion | B7.1, B7.3 | 13, 06, 05, 02 | after 13 (H and iH) |
| 16 | Graph cut + ROI cleanup | B7.4, B7.5 | 06; Q13 | now |
| 17 | Sulcal depth | B9.5 | 07 merged (VOXELISE) | after PR #6 merges |
| 18 | Statistics, legacy output view, MaCRUISE pipeline wiring | B10.1, B10.4, integration of B10.2/B10.3 | 08, 12–17 (stage bodies), 04, 05, 07 | stats and writers now on synthetic inputs; wiring last |
| 19 | PDF/QA layout | B12.3 | 09, 18 | after 18 |
| 20 | LGI on the ported TRACE (`lgi` driver) | B9.2–B9.4 follow-up | 07 merged, 04 | after PR #6 merges |
| 21 | HSD follow-up: replay BISCUIT's four-call chain | B6.5 follow-up | 05 | now |

Suggested order: wave A (independent files) **12, 16, 21, 13**; wave B **15, 17, 20**, and 14 after the owner answers Q3; wave C **18, 19**.

## 12 — MaCRUISE hemisphere separation (B5.2)

* **Scope.** `pybiscuit.surfprep.hemisphere`: orientation fix, cutting plane (spec §4.2 steps 1–2: 101 candidates, midpoint rule), boundary loops (step 4), island removal as "two largest components" (step 5),
  disk patch + harmonic fill (steps 6–8; own Delaunay of the same point set, not Qhull-identical), seam smoothing (step 9), left/right by native mean x (step 10), outputs of step 11 (`<h>.mask.txt`, `hemi_map.txt`, hemisphere VTKs).
  The plane selection standing in for `MeshContour` is a documented hypothesis (shortest cut) with the choice isolated in one function; a CLI `pybiscuit mesh-contour` mirrors the observed argv/stdout (spec §3 T39).
  Input frame: the native surface plus an alignment frame (legacy `aff.txt`/MNI copy, or a plane estimated from SEG_refine L/R labels — Q4).
* **Acceptance.** Synthetic: two ellipsoid-like hemispheres joined by a bridge → two closed, genus-0, consistently oriented meshes, mask 0 exactly on patch vertices, `hemi_map` round-trips original indices.
  Legacy sample (`~/dev/mipav/sample-macruise/output`): both hemispheres closed; left/right correct; patch size reported. Against the reference: `hemi_map` agreement ≥ 99 % of original vertices once an image run exists (B1.1); until then "provisional".
* **Risks.** `MeshContour` rule; `dmperm` order (spec §14 item 5); cyclic alignment of the patch boundary (item 6).

## 13 — MeshProperty / MeshDeform black-box port and feature maps (B3.1–B3.3, B9.1)

* **Scope.** `pybiscuit.mesh.property`: mean curvature (`-m`), shape index (`-s`), vertex area (`-v`) with `--surfsmoothing`, `--tensorsmoothing`, `-q` and the normalisation chain `--normbefore --normafter --prctile --datasmoothing --invert`, for exactly the argv combinations of spec §3 (T17, T19, T25, T26, T31, T38).
  `pybiscuit.mesh.deform`: `-t` translate and `--smooth N` only. The BISCUIT feature definitions built on them: registration `H = −4·H(white, 5, 5)`, `iH`, parcellation `−H` (spec §5.1, §6.2).
  Start from the ported MeshLib curvature code (`mesh/curvature.py`).
* **Acceptance.** E-class vs image goldens for each combination (image run, B1.4/Q1); before that, analytic tests (sphere: H = 1/R, shape index ±1, area sum) and a report that lists each flag's inferred semantics with the experiment that supports it.
* **Risks.** Flag semantics are only partly visible in `strings` (spec §3 note); needs probes on analytic surfaces with the image binary.

## 14 — FreeSurfer-replacement surface prep (B5.3–B5.7)

* **Scope.** Gated by the owner's answer to **Q3** (spec §4 shows pial and thickness come from FreeSurfer's intensity model, not MaCRUISE's outer surface). `pybiscuit.surfprep`: optional volume prep (spec §4.3–4.5 tables), white refinement + pial + thickness,
  mid surface (exact), `smoothwm` + inflation + `sulc`, sphere mapping (spec §4.6–4.8). Replacement class; the PyMaCRUISE outer-level-set pial is the default proposal.
* **Acceptance.** R-class (validation-tolerances.md): surface distances, `sulc/curv/iH` r ≥ 0.95 after registration, thickness r ≥ 0.9, no flipped faces, HSD energy within 5 %. Needs reference outputs from an image run on the legacy sample.
* **Risks.** Largest scientific risk of the project; clean-room rule for FreeSurfer (papers and black-box behaviour only). Split into two briefs if scope grows (surfaces; inflation + sphere).

## 15 — Parcellation features, alignment and fusion (B7.1, B7.3)

* **Scope.** `pybiscuit.parc.features` (spec §6.2: z-score, soft clamp ×2, `%f` rounding), alignment (spec §6.3: rigid SphericalRemesh deform at degree 0, `HSD -d 0 --noguess --icosahedron 5` against `parc/<p>/<h>.curv.mean.txt`),
  class↔label mapping from `label.csv` (spec §6.1), `parc.fusion.fuse` (spec §6.6), and a driver `parc.parcellate(...)` that chains features → network (brief 06) → graph cut/cleanup (brief 16) → fusion.
* **Acceptance.** Feature files vs image `.tmp` (E); network scores unchanged (brief 06); fusion on hand-built vote tables (unit) and on image outputs (E, ≥ 99.5 % vertices, Dice ≥ 0.98).
* **Risks.** The extra HSD call uses native-resolution spheres and `-o` = input (report H-8); CIVET degree 10 is out of scope for the MaCRUISE route.

## 16 — Graph cut and ROI cleanup (B7.4, B7.5)

* **Scope.** `pybiscuit.parc.graphcut` (spec §6.7) with an **independent, licence-clean** α-expansion (Q13): float32 capacities, Potts × edge length, upper-triangle edges, label order 0…L−1, stop when a cycle does not decrease the energy; `pybiscuit.parc.cleanup.roi_correction` (spec §6.8) with the tie rule and the empty-candidate case defined.
  Pure Python/Numba; no new dependencies unless SciPy's max-flow is adequate (integer capacities only → a small float max-flow is needed).
* **Acceptance.** Brute-force optimality on tiny graphs; energy never above the initial argmin labelling; monotone cycles; determinism; the cleanup's invariants (one component per label afterwards). Against the image: labels E (≥ 99.5 %) on the legacy sample. Ties explained (brief 03 helpers).
* **Risks.** BK max-flow tie behaviour is not reproducible without the library; equal-cost cuts may differ.

## 17 — Sulcal depth (B9.5)

* **Scope.** `pybiscuit.quant.sulcal_depth` (spec §9.2) reusing `quant/_voxelise.py` and the closing/morphology of brief 07 where the semantics coincide; SciPy for `distance_transform` replacements must reproduce `bwdistgeodesic` quasi-Euclidean (chamfer 1, √2, √3 through the mask) — implement as a masked Dijkstra/chamfer propagation, not a Euclidean transform.
  `pybiscuit sulcal-depth in.vtk out.txt`.
* **Acceptance.** Analytic: sphere with a slot → depth equals slot geometry within the voxel tolerance; monotone along a sulcus. Against the image: E (p99 ≤ 1e-4 relative on cortex; hull behaviour per spec §15 item 6). Provisional until an image run.
* **Risks.** `strel('ball')`, `interp3` with `Inf` neighbours, `bwperim` border handling — same unverified MATLAB behaviours as `OuterHull`.

## 18 — Statistics, legacy output view, pipeline wiring (B10.1, B10.4, integration)

* **Scope.** `pybiscuit.stats` (spec §10: ROI, curve, aggregates 20000–20012, CSV), registered-space map writer (SphericalRemesh onto `sphere_327680`), `legacy/` writer mirroring spec §13.1 including naming irregularities, and the MaCRUISE stage bodies registered on brief 08's graph
  (inputs/outputs per spec §3, §9.0). Stages whose ports are not merged stay `StageBlocked`.
* **Acceptance.** Statistics: hand-computed tiny cases (N−1 std, empty ROI → NaN, junction de-duplication, `w ≥ 0.5` rule); CSV column order; the legacy tree diff against spec §13.1 = ∅ for the names; end-to-end dry run on the legacy sample prints the plan with the blocked stages marked.
* **Risks.** CSV numeric format (R2012a `export`) unknown until the image run.

## 19 — PDF / QA layout (B12.3)

* **Scope.** `pybiscuit report`: pages reg / parc per protocol / curve / quant with the tile layouts, colour ranges, header block and page sizes of spec §11, on brief 09's renderer; merge order of the image wrapper. No LaTeX/MATLAB dependency.
* **Acceptance.** Owner visual review against `reference/vuiis_pdfs`; snapshot tests of layout geometry; page sizes 8.5 × {10.5, 8.7, 9.2, 11.7} in.

## 20 — LGI on the ported TRACE (follow-up to 07)

* **Scope.** `pybiscuit lgi` driver for BISCUIT's use (spec §9.3): `OuterHull` (brief 07) → hull correspondence (klaplace substitute or port, Q7) → TRACE curves (brief 04) → `Gyrification` with `-m 316 --speed 0.2 --refHullArea 77100`. Reports hull area on a real pial surface.
* **Acceptance.** End-to-end on the template `lh.white` and a pial-type surface; per-vertex agreement with the image when available; analytic diagnostics of brief 07.

## 21 — HSD follow-up: BISCUIT's four-call chain (follow-up to 05)

* **Scope.** Close report §3.2: tests that replay the four chained single-property calls (spec §5.3) with `-c C C --outputcoeff C C`, mesh/property length mismatch, `--icomesh` radius-100 spheres with the guess on, both `rh` stage-2 orders, and the parcellation call `-d 0 --noguess --icosahedron 5` with `-o` = input.
  A `pybiscuit register` convenience wrapper for the chain (inputs: `<h>.sphere.vtk`, `res.{iH,sulc,H}.txt`, template dir) is allowed, behind the existing CLI.
* **Acceptance.** Oracle runs of the same chains (non-FMA HSD build) agree with the port within the existing tolerances; the guess behaviour for the `--icomesh` configuration is recorded as a golden; image check queued (Q1).
