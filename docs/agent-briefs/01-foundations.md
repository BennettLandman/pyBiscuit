# Agent brief 01 — Foundations (B0, B2, B4)

**Written 2026-10-20.** Read [AGENTS.md](../../AGENTS.md) and [plan.md](../../plan.md)
first; this brief covers one working session and does not override them.

## Context you need

- Ilwoo Lyu has been asked for the unencrypted BISCUIT MATLAB source
  (`/opt/BISCUIT/`, CBT v1.3, June 2021). Until it arrives, **do not** work on
  anything that depends on the orchestrator's behavior: B1, B5, B7.1, B7.4–B7.5,
  B9.5, B10. If the code arrives mid-session, stop and report — do not
  start reverse-engineering it in this session.
- No Linux host is available yet, so there are no goldens from the image. This
  session's checks are against **files shipped in the image** (spheres,
  templates, icosphere operators), which are authoritative.
- Owner questions are still open ([open-questions.md](../open-questions.md)).
  Do not decide any of them; use the stated defaults only where this brief says so.

## Scope

| Row | Deliverable |
|---|---|
| B0.1 | `pyproject.toml`, `src/pybiscuit/` skeleton, `pybiscuit --version`, MIT `LICENSE`, `LICENSES/` with upstream notices |
| B0.2 | GitHub Actions: ruff + pytest on {3.10, 3.13} × {ubuntu, macos, windows} |
| B0.3 | `src/pybiscuit/assets.py`: registry of every file under `reference/sif_FC_v2.2/root/opt/biscuit/{reg,parc,sphere}` with size + SHA-256 (data only; no download code yet) |
| B0.4 | `tests/fixtures/` generators: UV sphere, ellipsoid, folded sphere `r = 1 + a·Y_lm`, icospheres 0–7, all seeded |
| B2.1 | `pybiscuit.io`: VTK legacy (ASCII + binary; POINTS/POLYGONS/POINT_DATA; float/double), FreeSurfer `.asc`, FS binary surf/curv (via NiBabel), GIfTI, per-vertex `.txt`, PyMaCRUISE VTK frame header |
| B2.2 | tkRAS ↔ scanner-RAS conversion for `.asc` given a NIfTI header |
| B2.3 | `pybiscuit.mesh` topology: edges, one-ring, face adjacency, boundary loops, components, Euler characteristic/genus, orientation check |
| B4.1 | Icosphere generator that reproduces the shipped spheres (see below) |
| B4.3 | Rebuild the MeshConv operators and compare with the shipped `icosphere_*.pkl` |

Out of scope: B2.4–B2.6 (geometry ported from MeshLib, AABB, geodesics), B4.2,
HSD, TRACE, and anything else. Note follow-ups in your report; don't start them.

## Environment notes (owner's Mac)

- Repo: `~/dev/biscuit/pyBiscuit`. Extracted image: `~/dev/biscuit/reference/sif_FC_v2.2/root/opt/`.
  Plaintext CTF members: `~/dev/biscuit/reference/ctf_plain/`. Upstream clones:
  `~/dev/biscuit/reference/repos/`. If `reference/` is missing, regenerate it
  with `scripts/extract_reference.py` (needs `unsquashfs`).
- A partial `.venv` (Python 3.13.5) exists in the repo; dependency install was
  interrupted. Reuse or recreate it; it is gitignored.
- In the Claude Code sandbox, pip fails SSL verification (OSStatus -26276) and
  git cannot write `.git/config`. Those commands must run outside the sandbox;
  everything else should stay inside it.
- No cmake or Homebrew network access; clang++ and Accelerate are available.
  Not needed for this brief.

## Step details and acceptance

### B0 — scaffold
- Python `>=3.10,<3.14`; setuptools src layout like PyMaCRUISE
  (`~/dev/mipav/pymacruise/pyproject.toml`). Core dependencies: numpy, scipy,
  nibabel, numba. Extras: `parc = [torch]`, `viz = [pyvista, matplotlib]`,
  `dev = [pytest, ruff, build]`. No other runtime dependencies without a
  justification line in the commit message.
- CLI: `pybiscuit --version` and `pybiscuit info` (prints versions plus the asset
  registry status). Subcommand stubs only for what is implemented.
- **Accept:** `pip install -e .[dev]` from a fresh venv; `pytest` and `ruff check`
  pass locally; CI green on the PR.

### B2.1 — I/O
- The VTK reader must load, without error and with the correct counts:
  - every `reference/sif_FC_v2.2/root/opt/biscuit/reg/*/?h.{white,inflated}.vtk`
    (163,842 vertices each, ASCII `float`),
  - `biscuit/sphere/sphere_{5120,20480,81920,327680}.vtk`,
  - `ctf_plain/SurfPostproc/common/parc/mesh_files/icosphere_{5,6}.vtk`,
  - a PyMaCRUISE surface, e.g.
    `~/dev/mipav/pymacruise/build/invivo-base-v3-dura002-no-regrid-2026-10-14/stages/surface-thickness/attempt-01/inner_surface.vtk`
    (read the frame from its header line `PyMaCRUISE coordinate_frame=RAS units=mm`).
- The `.asc` reader must load
  `~/dev/mipav/sample-macruise/output/Surfaces_FreeView/target_image_GMimg_innerSurf.asc`
  and `Surfaces_MNI/target_image_GMimg_innerSurf.asc`. Don't try to work out
  how BISCUIT relates them; that waits for the MATLAB source (Q4).
  *Correction (2026-10-20, after execution):* this brief originally gave the
  inner surface as 243,822 / 487,640 and called the two meshes independent.
  That count is the **central** surface. Both inner files are 250,874 / 501,744
  and are the same mesh related by a rigid transform; see
  [foundations report](../reports/foundations-2026-10-20.md).
- Per-vertex `.txt`: `reg/*/?h.*.{mean,var}.txt` must give 163,842 values.
- Round-trip tests (write → read → equal) for every format on the fixtures.
  Real-data checks are tests marked `@pytest.mark.reference` that skip when
  `reference/` or `~/dev/mipav` is absent. **Never copy real data into the repo.**

### B2.2 — frames
- Implement scanner RAS = `T1.affine @ inv(vox2ras_tkr) @ tkRAS` (derive
  `vox2ras_tkr` from the header with NiBabel's MGH/tkr helpers; do not consult
  FreeSurfer source).
- **Accept:** synthetic-header unit test. Diagnostic only: on the legacy sample,
  check that the converted inner surface lies within the brain mask of
  `sample-macruise/output/Raw/T1.nii.gz` / `SegRefine/SEG_refine.nii.gz`
  (fraction of vertices inside a dilated mask). Report the number.

### B2.3 — topology
- Pure NumPy/SciPy sparse; Numba only if profiling shows a need.
- **Accept:** χ = 2 and genus 0 for all shipped spheres and template white
  surfaces; boundary loops detected on a cut fixture; orientation check flags
  a flipped-face fixture.

### B4.1 — icospheres
- Reproduce the subdivision scheme so that the generated level-k icosphere
  matches `sphere_{5120,20480,81920,327680}.vtk` (ico 4–7) **vertex for vertex,
  in the same order**, and the same faces. If the order cannot be matched, find
  and document the permutation; that is a partial pass and must be reported.
- Check whether `reg/*/?h.white.vtk` (ico7, 163,842) share the face list of
  `sphere_327680.vtk`. If they do, the template sphere is that file. Record it.
- Compare also to `icosphere_{5,6}.vtk` (parc meshes) and to upstream
  `reference/repos/SphericalLabeling/gen_ico.py` (MIT), which may be the
  generator used. Vertex orders may differ between BISCUIT and UGSCNN meshes.
- **Accept:** max |Δxyz| ≤ 1e-6 (the files are ASCII floats), with identical faces.

### B4.3 — MeshConv operators
- **Security:** the `.pkl` files come from a third-party image. Do **not** call
  `pickle.load` on them. Inspect them first with `pickletools.dis` and load
  through a restricted `Unpickler` whose `find_class` allows only the
  numpy/scipy.sparse reconstructors you have seen in the disassembly.
- Identify the stored fields (UGSCNN typically stores V, F, Laplacian, gradient
  components, and the EW/NS fields). Regenerate them from your icospheres
  following `reference/repos/SphericalLabeling/meshcnn/` (MIT), and compare
  them field by field.
- Use the FC image's `.pkl` files (`reference/ctf_plain/`), not the cbt ones;
  they differ (see reference-artifacts.md).
- **Accept:** max abs diff ≤ 1e-6 per field (sparse: same sparsity pattern).
  Plan to ship regenerated operators (or `.npz`), never the original pickles.

## Evidence and hand-back

- Work on a branch `b0-b2-b4-foundations`; open a PR to `main`. Commit
  messages reference row IDs. Include the Co-Authored-By trailer the
  harness provides.
- For each row, update its Status cell in plan.md with: test names, the
  command, pass/fail counts, and the commit. Put measured numbers (mask
  fraction, `.asc` comparison, icosphere diffs, pkl diffs) in
  `docs/reports/foundations-YYYY-MM-DD.md`.
- Add a port note `docs/ports/icosphere.md` (scheme, ordering, which files it
  reproduces, deviations).
- Update the Status section at the top of plan.md.
- Final message to the owner: rows done or not done with reasons, any
  surprises (especially the native-vs-MNI `.asc` finding and the template
  sphere identity), and proposed next rows.

## Stop conditions

Stop and report rather than improvising if: a shipped file contradicts this
brief; matching a shipped file seems to need a guess about orchestrator
behavior; a new runtime dependency seems necessary; or the MATLAB source
arrives.
