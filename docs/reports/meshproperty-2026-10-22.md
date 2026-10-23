# Brief 13 Report — `MeshProperty` / `MeshDeform` Black-Box Port & Feature Maps (`B3.1–B3.3`, `B9.1`)

**Date:** 2026-10-22  
**Branch:** `brief-13-meshprop`  
**Port note:** [`docs/ports/meshproperty.md`](../ports/meshproperty.md)  
**Code:** `src/pybiscuit/mesh/property.py`, `src/pybiscuit/mesh/deform.py`, `src/pybiscuit/cli.py`, `src/pybiscuit/mesh/__init__.py`  
**Tests:** `tests/test_meshproperty.py`, `tests/test_meshdeform.py`

---

## 1. Executive Summary

- **Binary reverse-engineering completed (`B3.1–B3.3`):** Disassembled `/opt/biscuit/bin/MeshProperty`, `/opt/biscuit/bin/MeshDeform`, and `/opt/biscuit/bin/MeshContour` from `biscuit_FC_v2.2.sif` (`SHA-256 e70aea62…4329`) alongside `libMesh.so.1.0` (`MeshLib`). Every flag, formula, post-processing order, floating-point cast, and VTK/text output format is now exact and documented in [`docs/ports/meshproperty.md`](../ports/meshproperty.md).
- **100% byte-for-byte identity on all `golden-image` fixtures (`B3.2`, `B3.3`):**
  - `tests/data/golden-image/meshproperty/{all_maps,quad,smoothed}`: **15/15 output files** (`.C.txt`, `.S.txt`, `.H.txt`, `.K.txt`, `.VertArea.txt`, and `.property.vtk`) match the container outputs **byte-for-byte** (`max_abs` vs parsed float64 $\le 5.00 \times 10^{-7}$, i.e. half of the last printed `%f` digit; `0` differing lines).
  - `tests/data/golden-image/meshdeform/{translate,matrix,sphere1,sphere2}`: **4/4 `deformed.vtk` files** match the container outputs **byte-for-byte** (`max_abs = 0.0` for `translate` and `matrix`; `max_abs = 9.31 \times 10^{-10}` vs unrounded float32 and `0` differing lines in `%.9f` VTK output for `sphere1` and `sphere2`).
- **100% byte-for-byte identity on real FreeSurfer-route subject outputs (`B9.1`):**
  - Across all **3 subjects × 2 hemispheres** (`0050952`, `0050961`, `0050971` in `open-examples/`), `quant_mid_features(mid, curv_smoothness=2)` reproduces `quant/{lh,rh}.mid.{H,S,area}.txt` **byte-for-byte** (**0 differing lines out of 2,213,571 lines total**, `max_abs <= 5.00e-7`).
- **Three `orchestrator-spec.md` §15 open items resolved from binary disassembly:**
  - **§15 item 2 (`MeshDeform -t` sign):** `MeshDeform::translate` (`0x3b1f0`) **subtracts** the translation vector (`v[k] = float(double(v[k]) - t[k])`), and `MeshDeform::getCenter` (`0x3b570`, `-c`) prints the **negated** centroid `-cx -cy -cz`.
  - **§15 item 4 (`MeshContour` plane selection rule):** `MeshContour` (`0x1409`) slices the mesh at each of the 101 candidate offsets using `MeshLib`'s `Slicer` and selects the plane with the **minimum total polyline contour length** (confirming Hypothesis `H-MC1` / `"shortest"` in `pybiscuit.surfprep.contour`).
  - **§15 item 5 (`MeshProperty` post-processing & `--invert`):** `MeshProperty` applies `crop(prctile)` $\to$ `normalize(5.0)` (`--normbefore`) $\to$ `SurfaceUtil::smoothing(mesh, datasmoothing, data)` $\to$ `normalize(5.0)` (`--normafter`). `--invert` negates `m_cmin` and `m_cmax` in place without swapping them.

---

## 2. Validation Against Container Golden Fixtures (`tests/data/golden-image/`)

### 2.1 `MeshProperty` Golden Cases

| Case | CLI invocation | Output file | Vertices | `max_abs` vs parsed golden | Differing lines (`%f` / VTK) | Byte-identical? |
|---|---|---|---:|---:|---:|:---:|
| `all_maps` | `MeshProperty -i bumpy.vtk -o bumpy -c -s -k -m -v --writevtk` | `bumpy.C.txt` | 2,562 | `4.998e-07` | `0 / 2,562` | **Yes** |
| `all_maps` | (same) | `bumpy.S.txt` | 2,562 | `5.000e-07` | `0 / 2,562` | **Yes** |
| `all_maps` | (same) | `bumpy.H.txt` | 2,562 | `4.998e-07` | `0 / 2,562` | **Yes** |
| `all_maps` | (same) | `bumpy.K.txt` | 2,562 | `5.000e-07` | `0 / 2,562` | **Yes** |
| `all_maps` | (same) | `bumpy.VertArea.txt` | 2,562 | `4.995e-07` | `0 / 2,562` | **Yes** |
| `all_maps` | (same) | `bumpy.property.vtk` | 2,562 | `0.0` | `0 / 20,511` | **Yes** |
| `quad` | `MeshProperty -i bumpy.vtk -o bumpy -q -m -k --writevtk` | `bumpy.H.txt` | 2,562 | `5.000e-07` | `0 / 2,562` | **Yes** |
| `quad` | (same) | `bumpy.K.txt` | 2,562 | `4.991e-07` | `0 / 2,562` | **Yes** |
| `quad` | (same) | `bumpy.property.vtk` | 2,562 | `0.0` | `0 / 12,819` | **Yes** |
| `smoothed` | `MeshProperty -i bumpy.vtk -o bumpy -m -k --tensorsmoothing 2 --surfsmoothing 1 --datasmoothing 2 --writevtk` | `bumpy.H.txt` | 2,562 | `4.994e-07` | `0 / 2,562` | **Yes** |
| `smoothed` | (same) | `bumpy.K.txt` | 2,562 | `4.993e-07` | `0 / 2,562` | **Yes** |
| `smoothed` | (same) | `bumpy.property.vtk` | 2,562 | `0.0` | `0 / 12,819` | **Yes** |

### 2.2 `MeshDeform` Golden Cases

| Case | CLI invocation | Output file | Vertices | `max_abs` (vertices) | Differing lines (`%.9f` VTK) | Byte-identical? |
|---|---|---|---:|---:|---:|:---:|
| `translate` | `MeshDeform -i bumpy.vtk -o deformed.vtk -t 1.5 -2.0 0.5` | `deformed.vtk` | 2,562 | `0.000e+00` | `0 / 7,690` | **Yes** |
| `matrix` | `MeshDeform -i bumpy.vtk -o deformed.vtk --matrix 1 0 0 2 0 1 0 -1 0 0 1 0.5` | `deformed.vtk` | 2,562 | `0.000e+00` | `0 / 7,690` | **Yes** |
| `sphere1` | `MeshDeform -i bumpy.vtk -o deformed.vtk --sphere 1` | `deformed.vtk` | 2,562 | `9.313e-10` | `0 / 7,690` | **Yes** |
| `sphere2` | `MeshDeform -i bumpy.vtk -o deformed.vtk --sphere 2` | `deformed.vtk` | 2,562 | `9.313e-10` | `0 / 7,690` | **Yes** |

---

## 3. Validation Against Real Subject Runs (`open-examples/`, `B9.1`)

In `surf_quant_pipeline.m:13-22` (spec §9.1 T31), BISCUIT computes mid-surface mean curvature and shape index via:

```bash
MeshProperty -i <h>.mid.vtk -o <tmp3>/<h>.mid -s -m --surfsmoothing 2 --tensorsmoothing 2
```

and mid-surface vertex area (spec §5.5 T25) on the unsmoothed mid surface via:

```bash
MeshProperty -i <h>.mid.vtk -o <tmp1>/<h>.mid -v
```

Testing `quant_mid_features(mid, curv_smoothness=2)` against the container outputs in `<PATH>/open-examples/{0050952,0050961,0050971}/quant/`:

| Subject | Hemi | Vertices | `mid.H.txt` `max_abs` | `mid.H.txt` diff lines | `mid.S.txt` `max_abs` | `mid.S.txt` diff lines | `mid.area.txt` `max_abs` | `mid.area.txt` diff lines | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| `0050952` | `lh` | 137,067 | `5.000e-07` | `0 / 137,067` | `5.000e-07` | `0 / 137,067` | `5.000e-07` | `0 / 137,067` | **PASS (byte-identical)** |
| `0050952` | `rh` | 138,517 | `5.000e-07` | `0 / 138,517` | `5.000e-07` | `0 / 138,517` | `5.000e-07` | `0 / 138,517` | **PASS (byte-identical)** |
| `0050961` | `lh` | 101,784 | `5.000e-07` | `0 / 101,784` | `5.000e-07` | `0 / 101,784` | `5.000e-07` | `0 / 101,784` | **PASS (byte-identical)** |
| `0050961` | `rh` | 101,233 | `5.000e-07` | `0 / 101,233` | `5.000e-07` | `0 / 101,233` | `5.000e-07` | `0 / 101,233` | **PASS (byte-identical)** |
| `0050971` | `lh` | 129,055 | `5.000e-07` | `0 / 129,055` | `5.000e-07` | `0 / 129,055` | `5.000e-07` | `0 / 129,055` | **PASS (byte-identical)** |
| `0050971` | `rh` | 130,201 | `5.000e-07` | `0 / 130,201` | `5.000e-07` | `0 / 130,201` | `5.000e-07` | `0 / 130,201` | **PASS (byte-identical)** |

---

## 4. Key Binary Findings

1. **`--surfsmoothing` mutates `m_mesh` in place (`MeshProperty::init` at `0x3afe0`):** When `--surfsmoothing N` ($N > 0$) is passed, `SurfaceUtil::smoothing(m_mesh, N)` modifies `m_mesh` before principal curvatures, vertex areas (`-v`), data smoothing, or `--writevtk` mesh export run. In BISCUIT's pipeline, `T25` (`-v` for `mid.area.txt`) is a separate call without `--surfsmoothing`, whereas `T31` (`-s -m --surfsmoothing 2 --tensorsmoothing 2`) smooths the mesh twice before computing `H` and `S`.
2. **`MeshProperty::normalize(5.0)` (`0x3b4b0`) vs `extract_features.m` (`parc_normalize_feature`):** Both perform the same two-pass algorithm (pass 1: subtract mean, divide by std, soft-clamp outside $[-\theta, \theta]$ via $\pm\theta \pm (1 - \exp(\theta - |x|))$; pass 2: divide by std **without** subtracting the mean, then soft-clamp outside $[-\theta, \theta]$ again). `MeshProperty` uses population variance (`Statistics::var`, `ddof=0`) and $\theta = 5.0$, whereas MATLAB's `extract_features.m` uses `zscore`/`std` (`ddof=1`) and $\theta = 3.0$.
3. **`MeshDeform --sphere {1,2}` (`0x3b720`) pre-smooths 200 times:** Before calling `SurfaceUtil::sphere(m_mesh, sphere - 1, 10000)`, `MeshDeform::makeSphere` unconditionally calls `SurfaceUtil::smoothing(m_mesh, 200)`.
4. **`MeshDeform::applyMatrix` (`0x3b390`) mixed float32/float64 accumulation:** Each of the three matrix products per row converts the `float32` vertex coordinate to `double`, multiplies by the `double` matrix element, adds to the accumulator, and casts the partial sum to `float32` and back to `double` before adding the next column. Reproducing this exact cast sequence yields `max_abs = 0.0` against the container golden.
