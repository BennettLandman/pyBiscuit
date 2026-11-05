# Port note: parcellation network training (plan rows B7.6, B7.7)

Code: `src/pybiscuit/parc/train.py` (trainer, loader, metrics, CLI `pybiscuit parc-train`),
`src/pybiscuit/parc/gen_ico.py` (CLI `pybiscuit parc-gen-ico`), `save_safetensors` in
`src/pybiscuit/parc/checkpoint.py` (now shared by the converter and the trainer). Tests:
`tests/test_parc_train.py`. Needs the `[parc]` extra (torch, safetensors); nothing new is added to
the dependencies.

Reused, not duplicated: `SphericalUNet`/`MeshConv`/`ResBlock` (`unet.py`, port note
[parcellation-net](parcellation-net.md)), `canonicalize_state_dict` and the legacy-pickle audit
(`checkpoint.py`), `ParcModel` (`infer.py`), the operators (`sphere/operators.py`) and the UGSCNN
icosphere (`sphere/icosphere.py`, scheme `ugscnn`).

## Sources ported

ilwoolyu/SphericalLabeling @ `37fa7a0` (MIT), `reference/repos/SphericalLabeling`:

| File, lines | Ported to |
|---|---|
| `train.py` 22-25 `save_checkpoint` | `train.save_best` |
| `train.py` 27-71 `iou_score`, `accuracy`, `dice` | `train.class_scores` (one bincount pass, same integer counts) |
| `train.py` 73-100 `train` | `train.train_epoch` |
| `train.py` 103-144 `test` | `train.evaluate` |
| `train.py` 147-187 argument parser | `train.build_parser` (same flag names, types, defaults), `TrainConfig` |
| `train.py` 189-300 `main` (seeding, loaders, model, blackout, resume, optimiser, decay, epoch loop) | `train.train` / `train._train` |
| `loader.py` 1-131 `nv_sphere`, `S2D3DSegLoader` | `train.NV_SPHERE`, `train.SegDataset`, fold split `train.split_subjects` (49-76) |
| `train.sh` | `train.TRAIN_SH_DEFAULTS` (batch 4, 30 epochs, `max_level 5`, `feat 32`, lr 0.01, classes `0 1 2 3 4 6 7 8 10`, `in_ch curv iH sulc`, `deg 15`, seed 1, `drop 0`) |
| `gen_ico.py` 1-49, `meshcnn/mesh.py` 8-14 `export_spheres` | `gen_ico.write_vtk`, `gen_ico.export_spheres` |

`meshcnn/`: the trainer needs nothing that `unet.py` lacks. The stride-2 `MeshConv` branch is
still never instantiated (`model.py` builds only stride-1 `MeshConv` and `MeshConv_transpose`),
so it remains unported. Upstream `model.py` differs from the image's copy only by an unused
`nn.Dropout(p=0.2)` attribute (its calls are commented out; no parameters, no effect) and
`super().__init__()` spelling.

### Data augmentation (SphericalRemesh)

Training does depend on SphericalRemesh's augmentation mode: README step 2 runs
`SphericalRemesh -s <sphere> -r icosphere_5.vtk -c <coeff> -p <features> --outputProperty
<data>/features/<subj>.<h>.aug --deg0 0 --deg 15 --ftype double` (and the same with
`--ftype int16 --nneighbor` for labels), and `loader.py` reads the `aug0..aug<deg>` files it
writes. That mode (`-c` coefficients, the `--deg0..--deg` sweep that appends the degree to the
output prefix, `--ftype` binary `.dat` output, `--nneighbor` label transfer) is **already ported
and oracle-tested** in `sphere/remesh.py` (plan B4.2, [spherical-remesh](spherical-remesh.md),
cases `coeff_deg0/deg2`, sweeps, all `--ftype`). Nothing was added. Use
`pybiscuit spherical-remesh` with the README's flags to build a training set.

## Deviations (none changes what is learned from the same data and RNG stream, except 6)

1. **Device.** No `nn.DataParallel`; `data.cuda()` (which upstream calls even with `--no-cuda`)
   is `data.to(device)`. CPU is the tested path. State-dict keys therefore have no `module.`
   prefix; `canonicalize_state_dict` strips it from upstream files anyway.
2. **Operators / `--mesh_folder`.** Default `None`: the operators are regenerated
   (`build_operators`). If the folder is given, `icosphere_<level>.pkl` files are read with the
   restricted unpickler (`sphere.operators.safe_load_pickle`), never with `pickle.load`.
3. **`gen_ico`** writes only `icosphere_<level>.vtk` (upstream's main path writes only `.pkl`;
   its VTK writer runs only in the download fallback). The VTKs are what augmentation needs; the
   network does not read pickles. Vertex text is Python `str(float)` as upstream; the test checks
   the values equal `build_operators(level)['V']` exactly.
4. **Checkpoints.** Upstream writes one file, only when the validation Dice improves:
   `<log_dir>/checkpoint_latest.pth` + `_SUNet` + `_best.pth` (string concatenation; kept as
   `BEST_NAME`), holding `epoch, state_dict, dice, optimizer`. We write the same file (torch zip
   format, read back only with `weights_only=True`; the state dict has canonical keys and no
   sparse or `NS`/`EW` buffers) **and** `<BEST_NAME>.safetensors`, a `pybiscuit-parc-unet/1` file
   with `arch, epoch, dice, classes, in_ch, hemi, fold, kcv, seed` metadata that
   `ParcModel.from_safetensors` loads directly.
5. **Resume.** As upstream: `epoch` and `dice` are restored, weights are copied by name and unknown
   names are skipped (`load_my_state_dict`), the optimiser state is **not** restored. Loading goes
   through `torch.load(weights_only=True)`; legacy (non-zip) files are first audited by
   `audit_legacy_pickle`. Keys pass through `canonicalize_state_dict`, so upstream `.pth` files
   (aliases, `module.`, `NS`/`EW`) load too; `num_batches_tracked` is not copied (only used when
   BatchNorm momentum is `None`, which it is not). The shipped BISCUIT checkpoints have no
   `epoch`/`dice` and cannot be resumed (upstream: `KeyError`; here: `KeyError` with a message).
6. **RNG stream.** Seeding is upstream's (`torch.manual_seed(seed)` once, then model
   initialisation, then one sampler seed per epoch drawn by `DataLoader(shuffle=True)`), inside
   `torch.random.fork_rng` so the caller's global RNG is untouched. The subject shuffle uses
   `random.Random(seed)` (same sequence as `random.seed(seed); random.shuffle`). The parameter
   initialisation order follows our module construction order, which is not proven to match
   upstream's, so an upstream run and ours with the same seed are **not** claimed to start from the
   same weights. Runs of the port are bit-reproducible on CPU (`test_training_is_deterministic`).
7. **Learning-rate decay.** `StepLR(step_size=5, gamma=0.9).step(epoch)` (the deprecated
   explicit-epoch form) is applied as its closed form `lr0 * 0.9 ** (epoch // 5)` at the start of
   each epoch (`test_decay_schedule`).
8. **Blackout.** `keep_id` is a 1-D index (upstream: `np.argwhere` column, which adds a trailing
   axis to output and target); the selected elements, loss and metrics are the same.
9. **Loader paths.** Upstream splits the *whole path* on `.` (so a data folder containing a dot
   breaks it) and groups by the data folder's name; we parse the file name only. Missing channels
   or augmentations raise a `ValueError` naming the subject (upstream: a bare list/dict lookup error);
   a `.dat` with fewer than `nv` values raises instead of failing in `reshape`. `labels.astype(np.int)`
   (removed in NumPy 1.24) is `int64`. `--in_ch` is `required=True` (upstream crashes on `None`).
10. **Unchanged quirks kept:** `--test-batch-size` is parsed but unused (validation and test use
    `--batch-size`); labels outside `--classes` become class 0; `--drop` lists class *indices*;
    a class absent from both prediction and truth makes the mean NaN, and a NaN Dice never counts
    as best; validation/test use `aug0` only; the log lines (including "paramerters") are
    upstream's text, written to stdout and `<log_dir>/train.log`.

## Precision

Inputs: features float64 on disk, cast to float32; labels int16. Training is float32 throughout
(as upstream). Metrics are exact integer counts; the means are float64.

## Validation

Bit-exact validation is not possible for training. What is checked (`tests/test_parc_train.py`,
20 tests, ~7 s CPU):

* Loader against **upstream `loader.py` itself** (imported from `reference/repos`, NumPy 2 shim
  for `np.int`): identical file lists and identical `(data, labels)` items for train/val/test and
  folds 1, 4, 5 (the three branches of the split).
* Metrics against upstream `iou_score`/`accuracy`/`dice` (extracted from `train.py` with `ast`):
  identical counts, including dropped and absent classes.
* Parser: the set of option strings equals upstream's.
* Synthetic training (10 subjects, level-2 icosphere, 3 classes, `aug0..aug1`, `feat 4`): 6
  epochs run on CPU; mean training loss 1.060, 1.002, 0.941, 0.869, 0.784, 0.686 and validation
  Dice 0.11 → 0.89 (2026-11-04, torch 2.15.0, Python 3.13.5); two runs are bit-identical; resume
  continues at `epoch + 1`; decay schedule; CLI end to end.
* Round trip: the best `.safetensors` loads with `ParcModel.from_safetensors`, its metadata has the
  architecture, and `predict` equals the `.pth` weights in a fresh `SphericalUNet` exactly; the
  held-out subject is labelled > 50 % correct.
* `gen_ico`: vertices and faces equal the operator icosphere for levels 0-2.

**Intended acceptance test (row B7.7, not run):** retrain with `train.sh`'s settings on the labelled
data the shipped models were trained on (braincolor and DKT31, both hemispheres, the folds behind
`?h{,_2.._5}.pth`), then compare per-ROI Dice on held-out subjects with the 12 shipped checkpoints
run through `ParcModel`. Pass if mean Dice is within the fold-to-fold spread of the shipped models.
This needs the labelled training data and a licence check first ([open question Q18](../open-questions.md)).
No data was downloaded.
