# Report: Parcellation Pipeline Port and DKT31 Fusion (`B7.1`, `B7.3`)

**Author:** pyBiscuit porting team  
**Date:** 2026-10-22  
**Branch:** `brief-15-parc`  
**Plan rows:** `B7.1`, `B7.3`

## 1. Summary

This brief completes the cortical surface parcellation pipeline, connecting feature extraction, rigid spherical alignment, U-Net inference, clean-room graph-cut optimization, ROI connected-component cleanup, and DKT31 5-model fusion:

1. **Feature Extraction (`B7.1`)**:
   - `src/pybiscuit/parc/features.py`: implemented `normalize_parc_feature` and `extract_parc_features` matching `extract_features.m` and `spec §6.2`.
   - `load_protocol_labels`: parses `label.csv` for `lh` and `rh`, mapping class indices to protocol IDs with class 0 as `Unknown` (`0`).
2. **Multi-Model Fusion (`B7.3`)**:
   - `src/pybiscuit/parc/fusion.py`: implemented `refine_fold` and `fuse_folds` matching `parc.m:89-123` (spec §6.6).
   - Handles single-model paths (`braincolor`) and multi-fold softmax-weighted voting (`dkt31`).
3. **End-to-End Driver & CLI**:
   - `src/pybiscuit/parc/pipeline.py`: implemented `parcellate(...)` chaining all stages from native surfaces to final integer label files.
   - `src/pybiscuit/parc/cli.py`: implemented `pybiscuit parc` CLI subcommand.

## 2. Validation

- All unit tests in `tests/test_parc_core.py`, `tests/test_parc_features.py`, and `tests/test_parc_fusion.py` pass cleanly.
- Protocol labels verified against reference assets: 50 classes for BrainColor, 32 classes for DKT31.
- Pure Python/Numba execution: importing core modules never requires PyTorch.
- Full end-to-end driver tested with synthetic and reference fixtures.
