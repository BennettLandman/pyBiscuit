# Port note: parcellation network (plan rows B7.2, B7.2-oracle, B7.3)

Code: `src/pybiscuit/parc/{unet,checkpoint,infer,fusion}.py`, converter
`scripts/convert_parc_checkpoints.py`, converted-hash record
`src/pybiscuit/data/parc_converted_registry.json`. Oracle: `scripts/oracles/{build_parc_unet_oracle.sh,
parc_unet_oracle.py,run_parc_oracle.py,make_golden_parc.py,parc_acceptance_metrics.py}`. Tests:
`tests/test_parc_core.py` (no torch), `tests/test_parc_net.py` (committed tiny golden),
`tests/test_parc_reference.py` (12 checkpoints vs oracle; needs `reference/`). Report:
[parcellation-net-2026-10-20](../reports/parcellation-net-2026-10-20.md).

## Sources ported

Image CTF `SurfPostproc/common/parc/` (UGSCNN-derived, MIT; torch 1.4 era; sha256 of every staged file
in `oracles/parc-unet/stage.sha256`):

| File | Used for |
|---|---|
| `model.py` 1-73 | `SphericalUNet`, `Up`, `Down` -> `unet.py` |
| `meshcnn/ops.py` 1-164 | `MeshConv`, `MeshConv_transpose`, `ResBlock`, `DownSamp` -> `unet.py` |
| `meshcnn/utils.py` 9-27 | `sparse2tensor`, `spmatmul` -> `unet._csr`, `unet._spmm` |
| `loader_individual.py` | first `in_ch` channels, first `nv` vertices (`ParcModel.predict`); channel order `H, iH, sulc` (file suffix sort) |
| `test.py` 11-45 | eval-mode forward, `np.savetxt(fmt='%f')` per class, `SphericalRemesh -p ... -r osphere -s isphere --outputProperty` -> `infer.resample_to_native` |
| `test.sh` | `--feat 64 --in_ch 3 --min_level 0`, `--max_level` caller supplied |

Facts established from the files and checkpoints (all 12):

* The shipped operator pickles stop at level 5 (10,242 vertices), so `max_level = 5`. Both protocols
  use `fdim = 64`, `in_ch = 3`; `braincolor` has **50** output channels, `dkt31` **32**.
* `test.py` exports the **raw output of the last MeshConv** (no softmax, no argmax). Anything named
  "probability" downstream must apply its own normalisation; the orchestrator is encrypted (B7.1).
* The up-sampling `MeshConv_transpose` pads the coarse features with **ones** (not zeros) up to the
  finer level. Kept.
* Coarsening `ResBlock`: `conv1` -> keep first `nv_prev` vertices -> `bn1` ... and the shortcut is
  `conv_` -> slice -> `bn_`; slicing relies on the hierarchical vertex order of the icospheres.
* Stride 2 `MeshConv` is never instantiated by the network and is not ported.
* **DKT31 fusion is not in `test.py`** (one checkpoint per call). The five `?h{,_2.._5}.pth` are
  combined in `parc.m` (encrypted). `fusion.py` is an interface only: `fuse(outputs)` raises
  `NotImplementedError("... pending B7.1 ...")` unless the caller passes a rule.

## Deviations from the original (none changes results; verified bit-identical)

1. **Operators** come from `pybiscuit.sphere.operators.build_operators` (brief 01), converted to
   float32 CSR like the original `sparse2tensor` (COO float32); a `level -> dict` callable can replace
   the source (used to test the shipped pickles through the safe unpickler).
2. **Buffers**: `G, L, F2V, NS, EW` are non-persistent buffers. The checkpoints contain `NS`/`EW`
   (dense, float32) but not the sparse ones; `NS`/`EW` equal `float32(pickle NS/EW)` exactly and the
   regenerated ones to 5.7e-14 (float32 view), so the converter proves equality (< 1e-6) and drops
   them. `test.py` copied them over the pickle-derived buffers; with equal values that is a no-op.
3. **Layer registration**: the original registers each `ResBlock` layer as an attribute *and* inside
   `seq1`/`seq2`. The checkpoints (older code) list `conv1/conv2/conv3/conv_` as attributes and the
   batch norms only as `seq1.K`/`seq2.K`; shared storages (100 alias groups per file). Our modules hold
   each layer once under its attribute name; `canonicalize_state_dict` maps `seq1.{0,2,4,5,7,8}` (coarsening
   blocks) / `seq1.{0,1,3,4,6,7}` and `seq2.{0,1|2}` (others) to `conv1,bn1,conv2,bn2,conv3,bn3,conv_,bn_`,
   merging aliases (bit-equality enforced) and rejecting unknown positions.
4. **Tolerant loader**: `test.py`'s `load_my_state_dict` silently skips names it does not know. Ours
   loads strictly (the only absentee tolerated is BatchNorm `num_batches_tracked`, which legacy files
   predate and eval mode never reads).
5. **MeshConv weighted sum**: kept as the original broadcast multiply + two `torch.sum`s (a single
   matmul is ~3x faster but differs from the oracle by up to 1.4e-4 on scores of magnitude ~100, over
   the 1e-4 limit). Evaluated in blocks of 16 output channels to bound the `B x O x C x nv x 4` temporary
   (0.5 GB for the 50-class output layer otherwise); the chunked result is bit-identical to the oracle.
6. Pad tensor / `ReLU(inplace)`: functional `torch.relu`; same values.

## Output resampling (B7.3)

`resample_to_native(scores, isphere, osphere)` = `text_roundtrip` (what `np.savetxt('%f')` followed
by SphericalRemesh's `%f` read does: six decimals, float32) + `pybiscuit.sphere.remesh.resample`
(brief 02 port, barycentric). `isphere` must have exactly `nv = 10242` vertices (the original passes
the level-5 icosphere that carries the features); `quantize=False` skips the text round trip. The
original overwrites the per-class text files with the remeshed ones; we return arrays.

## Security (brief rules)

* Never `weights_only=False`. `checkpoint.audit_legacy_pickle` statically walks all five pickles of
  a legacy file with `pickletools.genops` (nothing executed) and refuses any `GLOBAL` outside
  `{torch._utils._rebuild_tensor_v2, torch.FloatStorage, collections.OrderedDict}` (the complete
  set found in all 12 files) and any `INST/OBJ/NEWOBJ/BUILD/STACK_GLOBAL/EXT*` opcode. Only then
  `torch.load(weights_only=True)`, which accepts the legacy format in torch 2.14 (verified), so the
  restricted-unpickler fallback of the brief was **not needed** and is not implemented.
* The source SHA-256 is checked against `asset_registry.json` before the file is opened for unpickling.
* The converted file is read back and compared bit for bit with what was converted. It carries a single
  metadata entry (source path/hash, architecture, drop list, redistribution notice). A single entry on
  purpose: safetensors writes the metadata map in hash order, which made the bytes (and SHA-256) vary
  between runs with several entries (caught by `test_legacy_pth_to_safetensors_roundtrip`).
* Runtime (`ParcModel.from_safetensors`) never touches `.pth`; it can pin the file hash.
* Converted weights stay out of git (`*.safetensors` is git-ignored) and out of any release until
  Q6/Q7 are answered. `parc_converted_registry.json` holds hashes only.

## Oracle (B7.2-oracle)

`build_parc_unet_oracle.sh` copies `ctf_plain/.../parc` byte for byte to
`~/dev/biscuit/oracles/parc-unet/stage` (hashes in `stage.sha256`) and makes a throwaway venv (torch
2.14.1, numpy 2.5.4, scipy 1.18.1, Python 3.13). `parc_unet_oracle.py` runs the unmodified `test.py`
with `runpy`; shims, all at import level:

1. `ops.pickle` replaced by a restricted loader (the six globals of brief 01); the stock
   `pickle.load` is never applied to the third-party `icosphere_*.pkl`. **Needed.**
2. `scipy.sparse.coo` alias if missing. *Not triggered* (still present in SciPy 1.18.1).
3. `torch.sparse.FloatTensor` fallback if missing. *Not triggered* (deprecated but present).
4. `np.savetxt` also writes `<file>.npy` (exact float32 output; the text has 6 decimals).
5. `os.system` copies the per-class text files to `raw/` before `SphericalRemesh` overwrites them,
   puts the non-FMA oracle binary on `PATH`, then runs the command unchanged.

`test.py` itself, `torch.load` (default `weights_only=True` in current torch) and the DataLoader are
unchanged. The oracle's `SphericalRemesh` is brief 02's `spherical-remesh-nofma` build.
Coverage check of the oracle's tolerant loader: every parameter and BN statistic is reached (the 160
dense names absent from the checkpoint are the `bn*` attribute aliases of `seq` entries, which are the
same modules); the 81 sparse buffers per model are built from the pickles.

Cases per checkpoint: `random` (seeded N(0,1), 3 x 10,242) and `template-<hemi>` (first 10,242 values
of the mb101 template `curv/iH/sulc` means for H/iH/sulc: real cortical fields). 24 oracle runs
(one network forward each). A third, committed, case is the tiny random-weight net
(`tests/data/golden/parc-unet/tiny.npz`, 0.34 MB, `max_level=3, fdim=8`, generated by the original
code; **test weights, not a BISCUIT model**).

## Known limits

* Feature extraction (which three channels, normalisation with `?h.curv.mean.txt`, resampling to
  the icosphere) is B7.1 and blocked; `predict` takes the three channels as given.
* Graph-cut, ROI cleanup and the DKT31 fusion rule are outside the network (B7.4, B7.5, fusion).
* CUDA is not supported (`cpuModel` path only, as `test.sh --no-cuda`).
