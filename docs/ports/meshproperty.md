# Port note: `MeshProperty`, `MeshDeform`, `MeshContour` & BISCUIT Curvature Feature Maps (`B3.1–B3.3`, `B9.1`)

- **Python modules:** `src/pybiscuit/mesh/property.py`, `src/pybiscuit/mesh/deform.py` (`src/pybiscuit/surfprep/contour.py` for `MeshContour`)
- **CLIs:** `pybiscuit mesh-property` (`MeshProperty v1.0`), `pybiscuit mesh-deform` (`MeshDeform v1.0`), `pybiscuit mesh-contour` (`MeshContour`)
- **Tests:** `tests/test_meshproperty.py`, `tests/test_meshdeform.py`
- **Validation report:** [`docs/reports/meshproperty-2026-10-22.md`](../reports/meshproperty-2026-10-22.md)
- **Reference binaries (`biscuit_FC_v2.2.sif`):** `/opt/biscuit/bin/MeshProperty`, `/opt/biscuit/bin/MeshDeform`, `/opt/biscuit/bin/MeshContour` (all dynamically linked against `libMesh.so.1.0`)

---

## 1. `MeshProperty` (`B3.1`, `B3.2`)

### 1.1 Binary Architecture & Method Map

Disassembly of `/opt/biscuit/bin/MeshProperty` (`0x7470` `main`, `0x3ae80–0x3c380` class methods) shows `MeshProperty` is a thin C++ wrapper over `MeshLib`'s `Mesh`, `SurfaceUtil`, and `Statistics` classes (`struct MeshProperty { double *m_cmin; double *m_cmax; double *m_data; Mesh *m_mesh; }`).

| C++ symbol (`MeshProperty`) | Address | Python port (`pybiscuit.mesh.property`) | Underlying `MeshLib` / numeric semantics |
|---|---|---|---|
| `MeshProperty::MeshProperty(const char*)` | `0x3ae80` | `read_vtk` + `_coerce_f32` | Calls `m_mesh->openFile(filename)` (`float32` vertices, `int32` faces, 1-ring connectivity). |
| `MeshProperty::init(int, int, bool, bool)` | `0x3afe0` | `mesh_property` | (1) If `surfsmoothing > 0`, calls `SurfaceUtil::smoothing(m_mesh, surfsmoothing)` in place (`smooth_vertices`). (2) Allocates `m_cmin`, `m_cmax`, `m_data` (`double[N]`). (3) Calls `SurfaceUtil::curvature_quad(m_mesh, m_cmin, m_cmax, NULL, NULL)` if `quad` (`-q`), else `SurfaceUtil::curvature(m_mesh, m_cmin, m_cmax, NULL, NULL, tensorsmoothing)`. (4) If `invert` (`--invert`), negates `m_cmin[i]` and `m_cmax[i]` in place (`xorpd -0.0`) **without** swapping `m_cmin` and `m_cmax`. |
| `MeshProperty::computeCurvedness` | `0x3bf20` | `curvedness_from_principal` + `postprocess_property` | $C_i = \sqrt{(c_{\min,i}^2 + c_{\max,i}^2) \cdot 0.5}$, then post-processing pipeline (§1.2). |
| `MeshProperty::computeShapeIndex` | `0x3be00` | `shape_index_from_principal` + `postprocess_property` | $S_i = \frac{2}{\pi}\arctan\!\left(\frac{c_{\max,i} + c_{\min,i}}{c_{\max,i} - c_{\min,i}}\right)$ with $\pi = 3.141592653589793$ (`0x3f440`), then post-processing pipeline (§1.2). |
| `MeshProperty::computeMeanCurvature` | `0x3c0a0` | `mean_curvature_from_principal` + `postprocess_property` | $H_i = (c_{\max,i} + c_{\min,i}) \cdot 0.5$, then post-processing pipeline (§1.2). |
| `MeshProperty::computeGaussianCurvature` | `0x3c220` | `gaussian_curvature_from_principal` + `postprocess_property` | $K_i = c_{\max,i} \cdot c_{\min,i}$, then post-processing pipeline (§1.2). |
| `MeshProperty::computeVertexArea` | `0x3b1c0` | `geometry.vertex_areas` | Barycentric 1-ring area $\frac{1}{3}\sum_{(j,j+1)} \frac{1}{2}\|e_1\|\|e_2\|\sin(\arccos(\hat{e}_1\cdot\hat{e}_2))$ evaluated on the (possibly `surfsmoothing`-smoothed) `float32` mesh; **not** passed through `crop`, `normbefore`, `datasmoothing`, or `normafter`. |
| `MeshProperty::crop(double)` | `0x3baf0` | `crop_percentile` | Percentile clamping (§1.2). |
| `MeshProperty::normalize(double)` | `0x3b4b0` | `soft_normalize` | Two-pass standardisation + exponential soft clamping at $\pm 5.0$ (§1.2). |
| `MeshProperty::saveFile(const char*)` | `0x3af50` | `write_property_txt` | Writes `m_data[i]` with `"%f\n"` (`6` decimal places). |
| `MeshProperty::saveSurface(const char*, int)` | `0x3b7b0` | `write_property_vtk` / `format_meshlib_vtk` | Calls `m_mesh->saveFile(fn, "vtk", false)` (`POINTS N float` with `%.9f`, `POLYGONS M 4M`, no normals), then appends `POINT_DATA N\nFIELD ScalarData K\n`. |
| `MeshProperty::saveSurface(const char*, const char*)` | `0x3b840` | `write_property_vtk` / `format_meshlib_vtk` | Appends `<fieldname> 1 <N> float\n`, each value as `"%f\n"`, and a trailing `"\n"`. |

### 1.2 Flag Semantics & Post-Processing Order

In `main` (`0x7470`), negative values of `--tensorsmoothing`, `--surfsmoothing`, and `--datasmoothing` are clamped to `0`. If `--writevtk` is enabled, `saveSurface(output + ".property.vtk", nFields)` writes the mesh header first, and then the five maps are evaluated and appended in the fixed order:

1. **Curvedness** (`-c` $\to$ `<output>.C.txt`, VTK field `Curvedness`)
2. **Shape Index** (`-s` $\to$ `<output>.S.txt`, VTK field `ShapeIndex`)
3. **Mean Curvature** (`-m` $\to$ `<output>.H.txt`, VTK field `MeanCurvature`)
4. **Gaussian Curvature** (`-k` $\to$ `<output>.K.txt`, VTK field `GaussianCurvature`)
5. **Vertex Area** (`-v` $\to$ `<output>.VertArea.txt`, VTK field `VertexArea`)

For each of the four curvature maps (`C`, `S`, `H`, `K`), post-processing runs in this exact sequence:

1. **`--prctile p` (`MeshProperty::crop(p)` at `0x3baf0`, when `p > 0.0`):**
   - If $p > 50.0$, replaces $p \leftarrow 100.0 - p$.
   - Converts $N$ to `float32` then `float64` (`cvtsi2ss` $\to$ `cvtss2sd`): $n_f = \text{double}(\text{float32}(N))$.
   - Computes truncated integer indices (`cvttsd2si`):
     $$i_{\text{lo}} = \lfloor n_f \cdot p / 100.0 \rfloor, \qquad i_{\text{hi}} = \lfloor n_f \cdot (100.0 - p) / 100.0 \rfloor.$$
   - Sorts a copy of `m_data` in ascending order (`std::sort`) and clamps `m_data` to $[\text{sorted}[i_{\text{lo}}],\, \text{sorted}[i_{\text{hi}}]]$.
2. **`--normbefore` (`MeshProperty::normalize(5.0)` at `0x3b4b0`):**
   - **Pass 1:** Computes population mean $\mu_1 = \text{Statistics::mean}(x, N)$ and population variance $\sigma_1^2 = \text{Statistics::var}(x, N) = \frac{1}{N}\sum_{i=0}^{N-1}(x_i - \mu_1)^2$ (`ddof=0`), standardises $z_i = (x_i - \mu_1) \cdot (1.0 / \sigma_1)$, and applies the continuous exponential soft clamp at threshold $\theta = 5.0$ (`0x3f448`):
     $$z_i \leftarrow \begin{cases} -\theta - (1 - \exp(z_i + \theta)) & z_i < -\theta, \\ \theta + (1 - \exp(\theta - z_i)) & z_i > \theta, \\ z_i & \text{otherwise}, \end{cases}$$
     which smoothly maps $(-\infty, -5) \to (-6, -5)$ and $(5, +\infty) \to (5, 6)$.
   - **Pass 2:** Recomputes $\sigma_2 = \sqrt{\text{Statistics::var}(z, N)}$, scales $w_i = z_i \cdot (1.0 / \sigma_2)$ (**without** subtracting the mean of $z$), and applies the same soft clamp at $\theta = 5.0$ a second time.
3. **`--datasmoothing K` (when `K > 0`):**
   - Calls `SurfaceUtil::smoothing(m_mesh, K, m_data)` (`smooth_data(faces, m_data, K)` on the 1-ring of `m_mesh`).
4. **`--normafter` (`MeshProperty::normalize(5.0)` at `0x3b4b0`):**
   - Runs the two-pass soft normalisation at $\theta = 5.0$ after data smoothing.

> **Subtlety of `--invert` on Shape Index (`-s`):** Because `MeshProperty::init` (`0x3b155–0x3b183`) negates `m_cmin` and `m_cmax` in place (`m_cmin[i] = -m_cmin[i]; m_cmax[i] = -m_cmax[i]`) without swapping them, the numerator $c_{\max} + c_{\min}$ and denominator $c_{\max} - c_{\min}$ in `computeShapeIndex` both flip sign, leaving `(cmax + cmin) / (cmax - cmin)` unchanged! Meanwhile `computeMeanCurvature` flips sign as expected.

---

## 2. BISCUIT Pipeline Invocations of `MeshProperty` (`B3.1`, `B9.1`)

All six `MeshProperty` call sites in BISCUIT ([`orchestrator-spec.md`](../reference/orchestrator-spec.md) §3) and their corresponding helpers in `pybiscuit.mesh.property`:

| Spec ID | Pipeline stage & MATLAB file | Exact `MeshProperty` flags | Post-processing in MATLAB | Python helper |
|---|---|---|---|---|
| **T17** | Registration (`surf_reg_macruise.m:24-29`, `surf_reg_civet.m`) | `-i <h>.white.vtk -o <h> -m --surfsmoothing 5 --tensorsmoothing 5` | Reads `<h>.H.txt`, multiplies by `-4`, overwrites `<h>.H.txt` (`%f\n`) | `reg_white_mean_curvature` |
| **T19** | Registration & FS prep (`surf_reg_macruise.m:39-44`, `run_postproc_fs_pipeline.m:325-332`) | `-i <h>.inflated.vtk -o <h>.inflated --normbefore --normafter --prctile 0.1 -m --datasmoothing 25 --invert -q` | Renamed to `<h>.iH.txt` (or converted to FreeSurfer curv `<h>.inflated.H`) | `reg_inflated_mean_curvature` |
| **T25** | Quantification vertex areas (`surf_reg_macruise.m:216-237`, `run_postproc_fs_pipeline.m`) | `-i <h>.mid.vtk -o <h>.mid -v` (and on `<h>.mid.reg.vtk`) | Copied to `quant/<h>.mid.area.txt` and `stat/<h>.mid.reg.area.txt` | `quant_mid_features` / `mesh_property(..., vertex_area=True)` |
| **T26** | Parcellation features (`extract_features.m:6-28`) | `-i <h>.white.vtk -o <tmp> -m --surfsmoothing 5 --tensorsmoothing 5` | `H = -H`, then two-pass `zscore`/`std` (`ddof=1`) + soft clamp at $\pm 3.0$ | `parc_white_mean_curvature` + `parc_normalize_feature` |
| **T31** | Quantification mid-surface curvature (`surf_quant_pipeline.m:13-22`) | `-i <h>.mid.vtk -o <tmp3>/<h>.mid -s -m --surfsmoothing N --tensorsmoothing N` ($N = \text{curv\_smoothness}$, default `2`) | Copied to `quant/<h>.mid.H.txt` and `quant/<h>.mid.S.txt` | `quant_mid_features` |
| **T38** | Curve PDF figure colouring (`curve_extract_pipeline.m:55-61`) | `-i <h>.inflated.vtk -o <tmp> -m -q --datasmoothing 10 --invert` | Used only to colour `curve.pdf`, then deleted | `mesh_property(..., mean_curvature=True, quad=True, data_smoothing=10, invert=True)` |

---

## 3. `MeshDeform` (`B3.2`, `B3.3`, `B5.6`)

Disassembly of `/opt/biscuit/bin/MeshDeform` (`0x70f0` `main`, `0x3b040–0x3b8b0` methods) shows `MeshDeform` applies up to nine in-place operations on `m_mesh` in a fixed order before writing `--output`:

| Order | CLI flag | C++ method (`MeshDeform`) | Python port (`pybiscuit.mesh.deform`) | Exact binary semantics |
|---|---|---|---|---|
| 1 | `--matrix m0..m11` | `applyMatrix` (`0x3b390`) | `apply_matrix` | Row-major $3\times 4$ `double` matrix. For each row $r \in \{0,1,2\}$, accumulates $a_0 = \text{float32}(0.0 + v_x m_{r,0})$, $a_1 = \text{float32}(\text{double}(a_0) + v_y m_{r,1})$, $a_2 = \text{float32}(\text{double}(a_1) + v_z m_{r,2})$, and stores $v'_r = \text{float32}(\text{double}(a_2) + m_{r,3})$. |
| 2 | `-t, --translate tx ty tz` | `translate` (`0x3b1f0`) | `translate_vertices` | **Subtracts** `(tx, ty, tz)` from each vertex (`subsd`): $v'_k = \text{float32}(\text{double}(v_k) - t_k)$. Resolves `orchestrator-spec.md` §15 item 2. |
| 3 | `--translateCentered tx ty tz` | `translateCentered` (`0x3b100`) | `translate_centered_vertices` | Calls `m_mesh->centering()`, then subtracts `(tx, ty, tz)` as in `translate`. |
| 4 | `-c, --center` | `getCenter` (`0x3b570`) | `negated_center` | Sums vertices in `float32`, divides by `float32(N)`, negates (`xorps -0.0f`), and prints `-cx -cy -cz\n` to stdout. |
| 5 | `--scale s` | `scale` (`0x3b710`) | `scale_vertices` | Calls `m_mesh->scaling(float(s))` (multiplies vertices by `float32(s)` if `s != 0`). |
| 6 | `--invertFace` | `invertFace` (`0x3b2e0`) | `invert_faces` | Reverses each triangle's vertex indices `(f[2], f[1], f[0])`. |
| 7 | `--sphere {1,2}` | `makeSphere` (`0x3b720`) | `make_sphere` | Calls `SurfaceUtil::smoothing(m_mesh, 200)` **first**, then `SurfaceUtil::sphere(m_mesh, sphere - 1, 10000)` (`1` $\to$ area preservation `type=0`, `2` $\to$ conformal `type=1`). |
| 8 | `--smooth K` | `smooth` (`0x3b8a0`) | `geometry.smooth_vertices` | Calls `SurfaceUtil::smoothing(m_mesh, K)` when `K > 0`. |
| 9 | `--noise sigma` | `addNoise` (`0x3b760`) | `add_gaussian_noise` | Calls `m_mesh->updateNormal()`, then displaces each vertex along its unit normal by $\mathcal{N}(0, \sigma^2)$ (`std::normal_distribution<float>(0, sigma)`). |
| 10 | `-o, --output path` | `saveFile` (`0x3b0e0`) | `format_meshlib_vtk` | If `output` is non-empty, calls `m_mesh->saveFile(output, "vtk", false)` (`%.9f` ASCII `float` points, no normals). |

---

## 4. `MeshContour` Plane Selection Rule (`B3.3`, `orchestrator-spec.md` §15 item 4)

Disassembly of `/opt/biscuit/bin/MeshContour` (`0x1409` `main`) resolves Hypothesis **H-MC1** from [`docs/ports/hemisphere.md`](hemisphere.md):

1. `MeshContour` constructs `Mesh mesh; mesh.openFile(input); Slicer slicer(&mesh);` (`libMesh.so.1.0`).
2. Initializes `min_len = FLT_MAX` (`0x7f7fffff` at `0x229c`) and `best_idx = -1`.
3. Loops over candidate plane constants $d_i$ (`i = 0 .. constants.size() - 1`):
   - Calls `n_groups = slicer.slicing(nx, ny, nz, d_i)`.
   - Sets `len = 0.0f`. For each contour group `g = 0 .. n_groups - 1`, retrieves the ordered intersection points `plist[0 .. sz-1]` via `Slicer::getSlice(plist, g)` and accumulates the open polyline length in `float32`:
     $$\text{len} \mathrel{+}= \sum_{j=1}^{\text{sz}-1} \|\text{plist}[j] - \text{plist}[j-1]\|_2.$$
   - If `min_len > len` (`comiss -0xdc(%rbp), %xmm0`), updates `min_len = len` and `best_idx = i` (strict `>`, so ties keep the lowest candidate index `i`).
4. Re-slices at `constants[best_idx]`, calls `Slicer::getSliceFaceIdx` for each group `g`, prints each cut face index (0-based, one per line), and finally prints `best_idx` on the last line.

This confirms that `DEFAULT_RULE = "shortest"` in `pybiscuit.surfprep.contour` (ranking candidate planes by minimum total intersection contour length, breaking ties toward the lowest candidate index) matches the `MeshContour` binary.
