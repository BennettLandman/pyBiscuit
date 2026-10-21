# Report: parcellation network (brief 06; rows B7.2, B7.2-oracle, B7.3)

Date 2026-10-20. Branch `brief-06-parcellation-net`. Port note:
[parcellation-net.md](../ports/parcellation-net.md). Environment: macOS arm64, Python 3.13.5,
torch 2.14.1 (CPU), safetensors 0.8.0, numpy 2.5.4, scipy 1.18.1.

## Result

* **B7.2 Done, oracle-provisional.** `pybiscuit.parc.unet.SphericalUNet` reproduces the image's original
  `model.py`/`meshcnn` code **bit for bit** (max abs diff of the per-class scores 0.0, argmax agreement
  100 %) on all 12 checkpoints, 2 input cases each (24 comparisons). Acceptance was <= 1e-4 / >= 99.9 %.
* **Operator source**: regenerated (brief 01) vs the shipped pickles read with the safe unpickler:
  max output difference 0.0 (limit 1e-6), all 24 cases (`parc_acceptance_metrics.py`) and the three
  checkpoints in `test_operator_source_makes_no_difference`.
* **B7.3 Partial.** Resampling to the native sphere is done and validated; the DKT31 five-model fusion
  is **not visible in `test.py`** (one checkpoint per call, raw scores out), so `parc/fusion.py` is an
  interface that raises `NotImplementedError("... pending B7.1 ...")` until a rule is supplied.
* **Inference time** (braincolor/lh, 50 classes, level 5, 10,242 vertices, M-series arm64 CPU):
  0.60 s per hemisphere with 10 threads, 2.15 s with 1 thread; model build + safetensors load 0.19 s.
  The original `test.py` run end to end (process start, model build from pickles, forward, 50 text
  files, 50-property SphericalRemesh) takes 4.6-6.8 s.

## Acceptance table (`scripts/oracles/parc_acceptance_metrics.py`)

Scores: our `ParcModel.predict` vs the oracle's exact float32 output. Native: our
`resample_to_native(...)` (six-decimal text round trip + brief 02 port) vs the oracle's
`SphericalRemesh` output text on `icosphere_5 -> icosphere_6`.

| Group | Checkpoints | Cases | Max abs diff (scores) | Argmax agreement | Max abs diff (native) | Native argmax agreement |
|---|---|---|---|---|---|---|
| braincolor (50 classes) | `lh`, `rh` | random, template | 0.0 | 100 % | 5.0e-7 | 100 % |
| dkt31 (32 classes) | `lh`, `lh_2..5`, `rh`, `rh_2..5` | random, template | 0.0 | 100 % | 5.0e-7 | 100 % |

Full per-row table: run the script. Score magnitude reaches ~104, so the exact match is meaningful
(a one-matmul formulation of the MeshConv sum differed by 1.0e-4 / 1.4e-4 on the same inputs and was
rejected; see port note, deviation 5). The 5e-7 native difference is the sixth-decimal text format of
the oracle's output file (half a unit in the last place).

Conversion: 12/12 checkpoints; source SHA-256 equal to `asset_registry.json`; static audit found exactly
`torch._utils._rebuild_tensor_v2`, `torch.FloatStorage`, `collections.OrderedDict` (once per file) and no
forbidden opcode; `torch.load(weights_only=True)` loads the legacy format directly (no fallback unpickler
needed); 254 canonical tensors per file (388 names listed: 80 layer aliases merged, 54 `NS`/`EW` names = 34
tensors dropped after equality with regenerated operators, max diff 5.9e-8 = float32 rounding of
the float64 operator); read
back bit-identical; output deterministic (second conversion produced the same SHA-256, enforced by a
test). Converted hashes: `src/pybiscuit/data/parc_converted_registry.json`.

## Tests

* `tests/test_parc_core.py` (core CI job, no torch): audit accepts allowed globals, refuses an
  `os.system` payload *without executing it* (marker file check), refuses protocol-4 `STACK_GLOBAL` and
  zip-format files; fusion interface; importing `pybiscuit.parc`/`checkpoint`/`fusion` does not import torch.
* `tests/test_parc_net.py` (`[parc]` job): tiny random-weight golden from the original code
  (key canonicalisation incl. alias/operator rejection, network vs oracle, legacy `.pth` -> safetensors ->
  inference round trip with a real legacy-format file written by `torch.save(...,
  _use_new_zipfile_serialization=False)`, determinism, hash pinning, strict loading, input truncation).
* `tests/test_parc_reference.py` (`reference` marker): everything in the acceptance table plus the
  committed-registry check. Needs `~/dev/biscuit/oracles/parc-unet/golden` (not in git, ~0.3 GB of
  outputs; regenerate with `scripts/oracles/run_parc_oracle.py`, 24 runs, ~3 min).

Full suite: **411 passed, 14 skipped** with `[parc]` installed (3 min); in a fresh venv with only
`pip install -e .[dev]` (no torch): **383 passed, 16 skipped** (the torch-dependent parc tests skip via
`importorskip`), so the core install and CI job are unaffected. `ruff check` clean.

## Requests and hand-back items (shared files and decisions are not mine)

1. **Merge / shared-file touches (small, conflict-prone only trivially):** `.github/workflows/ci.yml`
   (new `parc` job appended; the existing job does not install torch), `.gitignore` (`*.safetensors`),
   `pyproject.toml` (`parc = ["torch>=2", "safetensors>=0.4"]`; **safetensors is Apache-2.0**, needed
   to read/write the converted checkpoints; torch is BSD-3), `plan.md` rows B7.2 and B7.3 only.
2. **Asset registry:** converted-weight hashes live in my own `parc_converted_registry.json`, not in
   `asset_registry.json`, because `assets.verify(root)` expects every registered path under the image
   layout. Owner may merge the two schemas when asset hosting (B13.1) is designed.
3. **`tests/conftest.py` request:** `REFERENCE = REPO.parent / "reference"` breaks in a git worktree
   (`REPO` becomes `.claude/worktrees/<name>`, so every `reference` test skips). I used a symlink
   `.claude/worktrees/reference -> ~/dev/biscuit/reference` (untracked). A fix such as
   `REFERENCE = Path(os.environ.get("BISCUIT_REFERENCE", REPO.parent / "reference"))` would help all
   parallel sessions.
4. **Owner decisions / blockers:** (a) DKT31 fusion rule needs Lyu's MATLAB source (Q7) or an image
   run with intermediates (B1); (b) the converted weights must not be published until Q6/Q7 are
   answered (they are git-ignored; nothing was uploaded anywhere); (c) `docs/agent-briefs/README.md`
   status column and the top Status section of `plan.md` were left for the coordinating session, per the
   parallel rules.
5. **Not done (out of scope or blocked):** feature extraction (B7.1), graph cut (B7.4), ROI cleanup
   (B7.5), CUDA, a `pybiscuit parc` CLI subcommand (no pipeline consumer until B7.1).
6. **Provisional status:** the oracle is the *original Python code* under torch 2.14.1 on arm64, not the
   image run on Linux x86-64 (Q1). The image used torch 1.4; float32 sparse/dense reductions could differ in
   the last bits there. Row stays "oracle-provisional" until the B1 image goldens exist
   (`test.sh` goldens with intermediates are the right source for B7.1 features too).
