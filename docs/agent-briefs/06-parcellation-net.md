# Agent brief 06 — Parcellation network inference (B7.2, B7.3)

**Written 2026-10-20.** Runs in parallel with briefs 04, 05, 07 after brief 03
merges. Read [AGENTS.md](../../AGENTS.md), [plan.md](../../plan.md),
[parallel rules](README.md#parallel-sessions), and
`docs/ports/meshconv-operators.md`.

## Context

- The network code is plaintext in the image CTF:
  `reference/ctf_plain/SurfPostproc/common/parc/{model,test,loader_individual,cpuModel}.py`,
  `meshcnn/*.py`, `test.sh` (FC versions; the cbt versions differ).
  `test.sh` runs: `--feat 64 --in_ch 3 --min_level 0 --no-cuda`, plus
  `--max_level`, `--out_ch`, `--isphere`, and `--osphere` from the caller.
- Checkpoints: `reference/sif_FC_v2.2/root/opt/biscuit/parc/{braincolor,dkt31}/?h*.pth`
  (85 MB each; dkt31 has 5 per hemisphere). They are **legacy (non-zip) torch
  pickles** (magic `\x80\x02\x8a\x0a`).
- Which three input channels BISCUIT feeds and how they are normalised
  (`extract_features.m`, `parc/<p>/?h.curv.mean.txt`) is hidden in the
  orchestrator (B7.1, blocked). This brief covers the network, given a
  feature array.

## Security

The `.pth` files are third-party pickles. Never call `torch.load` with
`weights_only=False`. Try `weights_only=True` first. If the legacy format
refuses, load through a restricted unpickler that allows only the
tensor-rebuild globals you have listed with `pickletools`, as brief 01 did
for the operator `.pkl` files. Convert each checkpoint once to
**safetensors**, record SHA-256 in the asset registry, and only ever load
the safetensors at runtime.

## Scope

| Row | Deliverable |
|---|---|
| B7.2 | `pybiscuit.parc.unet`: SphericalUNet + MeshConv in current PyTorch (CPU), using brief 01's regenerated operators, not the pickles. Checkpoint conversion script `scripts/convert_parc_checkpoints.py`. |
| B7.2-oracle | **Oracle**: the image's original `model.py`/`meshcnn` code run unchanged (except for import shims) under current torch in a throwaway venv, fed identical inputs. Record any shim. |
| B7.3 | Per-class output resampling to the native sphere via brief 02's SphericalRemesh port (the `--outputProperty` path in `test.py`). DKT31 5-model fusion: implement the rule from `test.py` if it is visible there; otherwise leave it as an interface with the rule marked "pending B7.1". |
| — | Licence: UGSCNN MIT (done in brief 02); SphericalLabeling MIT. Model-weight redistribution is still Q6/Q7. Do not publish converted weights anywhere. |

## Acceptance

- Our network vs the oracle code on identical random and fixture inputs, for
  all 12 checkpoints: per-class output max abs diff ≤ 1e-4; argmax agreement
  ≥ 99.9 % of vertices.
- The operator source (regenerated vs shipped pickle) makes no output
  difference beyond 1e-6.
- Inference time per hemisphere, CPU: report it.
- Optional `[parc]` extra only: the core install and CI without torch must
  still pass. Add one CI job with torch CPU wheels.
