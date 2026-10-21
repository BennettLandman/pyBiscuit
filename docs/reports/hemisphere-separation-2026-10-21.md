# Report: MaCRUISE hemisphere separation (brief 12, plan row B5.2)

Date 2026-10-21. Branch `brief-12-hemisphere`. Status of row B5.2: **Done, oracle-provisional** (no image run yet, Q1; the `hemi_map` >= 99 % criterion against the reference cannot be checked until one exists). Port note: [hemisphere.md](../ports/hemisphere.md).

## What was built

* `pybiscuit.surfprep.hemisphere`: `separate_hemispheres(mesh, reference)` implements spec §4.2 steps 1 and 3-11; `write_outputs` writes `<h>.target_image_GMimg_innerSurf.vtk` (MATLAB `write_vtk` format), `<h>.mask.txt` (1 original, 0 patch) and `hemi_map.txt` (three 0-based ints per surviving original vertex).
* `pybiscuit.surfprep.contour`: the 101 candidate offsets (step 2, ported) and the **isolated MeshContour hypothesis** (`select_plane`, `PLANE_RULES`: `shortest` = H-MC1 default, `fewest_faces` = H-MC2).
* `pybiscuit.surfprep.frames`: alignment frames (MNI copy, `aff.txt`, SEG_refine label plane fit, Q4 default).
* CLIs: `pybiscuit mesh-contour <mesh> nx ny nz d1,d2,...,` (prints cut face ids then the chosen index, as `hemi_separation.m` parses it) and `pybiscuit hemi-split --surf ... [--ref-surf | --aff | --seg] --out DIR` (also writes `hemisphere.json` with the plane, loops, patch sizes). One block in `cli.py`.
* Tests: 13 synthetic/unit tests (`tests/test_hemisphere.py`), 5 legacy-sample tests (`tests/test_hemisphere_legacy.py`, marker `reference`). Full suite: 539 passed, 22 skipped; `ruff check .` clean.

## Acceptance

**Synthetic** (bridged dumbbell icosphere, `tests/fixtures/bridged.py`, level 4 and 5): both hemispheres closed, manifold, genus 0, one component, consistently oriented; mask 0 exactly on the interior patch vertices; `hemi_map` is in whole-brain order and indexes the right hemisphere vertices, and > 80 % of the mapped vertices keep their exact coordinates (only the seam neighbourhood moves); `lh` has the smaller x. A 20 degree rotated head split correctly with the aligned reference copy.

**Legacy sample** (`~/dev/mipav/sample-macruise/output`, 250,874 vertices, **provisional**), frame from `aff.txt` (identical to the MNI copy) and from labels:

| Frame | plane (offset mm) | loops (vertices) | island vertices | lh / rh vertices | `mask == 0` lh / rh | mapped of 250,874 |
|---|---|---|---|---|---|---|
| `aff.txt` / MNI copy | -1.214 (candidate 0 of 101) | 223, 227, 28 | 49 | 129,122 / 129,481 | 5,202 / 2,601 | 250,800 |
| SEG_refine labels | -0.501 (candidate 0) | 225, 224, 6 | 8 | 129,186 / 129,458 | 5,202 / 2,601 | 250,841 |

Both hemispheres are closed, genus 0, consistently oriented; `lh` has mean x -26.4 mm, `rh` +26.2 mm. Patch size: three patches in total (the two bridge loops plus one small hole); each carries 2,601 interior vertices because the patch grid has a fixed 51x51 size (as in MATLAB), so `mask == 0` totals 7,803 vertices. The label-fitted midline normal is 0.84 degrees from the MNI x axis; the label frame and the MNI frame agree on the hemisphere of 99.985 % of the vertices both keep (38 mismatches of 250,8xx). Run time about 5 s.

## Findings

1. **MeshContour rule is not identifiable from the data we have.** On the legacy sample the cut length rises almost monotonically across the 101 candidates (523 to 560 faces), so both hypotheses pick candidate 0, the extreme of the range. Forcing candidates 25/50/75/100 changes the hemisphere assignment of 49/133/217/297 vertices (>= 99.88 % agreement), so the downstream effect of a wrong rule is small, well inside the 99 % target. A black-box probe of the image binary (spec §15 item 4) is still needed to settle it. That the pick sits at the extreme is itself a warning that H-MC1 may not be the real rule.
2. **The legacy sample has a second, thin connection at the plane** (around y = -68 mm native, loops of 28 and 20 vertices) and a 49-vertex knob that the plane cuts off. MATLAB's `getBoundary` takes exactly two loops, so it would not close this sample as written; either the real `MeshContour` returns a different face set or the 2022 source differs from the data's producer. Deviation D4 in the port note: all loops are walked, cut-off pieces (not among the two largest components) are dropped with their loops, the remaining holes are patched. Cost: 74 vertices of 250,874 are absent from `hemi_map` (bridge band, knob).
3. `Surfaces_MNI/aff.txt` maps **MNI to native** (the inverse reproduces the MNI `.asc` to 5e-6 mm). Q4 can be answered: the label plane fit replaces the MNI input with 0.84 degrees of error, which moves the split by 38 vertices.
4. Orientation: the legacy mesh is already outward oriented (no face reversal).

## Open items for the owner / follow-up

* Image run (B1.1) to check `hemi_map` agreement >= 99 % and to probe `MeshContour`; until then every number here is provisional.
* Requests, not edits: none for shared modules (only `cli.py` block added). `mesh/topology.py` was used read-only.
* Brief 14 (surface prep) consumes `separate_hemispheres`; the output vertices are already in the native frame, the FreeSurfer-frame shift (`MeshDeform -t`, spec §4.4) is that brief's concern.
* Patch cost: tiny holes get a 2,601-vertex grid. Faithful, but a proposal (patch resolution proportional to loop size) could go to `docs/proposals/` if the owner wants smaller `mask == 0` regions.

## Plan rows touched

B5.2 only (Status cell).
