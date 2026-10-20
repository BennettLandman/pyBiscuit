### B4.1 icosphere (measured 2026-10-20)

Generator: `pybiscuit.sphere.icosphere.icosphere(level, scheme)`; tests `tests/test_icosphere.py`
(`-m reference` tests skip when `reference/` is absent).

| Shipped file | scheme | faces identical | max abs diff, raw gen*R vs file | after `quantize_shipped` | 1-step tie flips | non-flip max diff |
|---|---|---|---|---|---|---|
| sphere_5120 (L4, R=100) | biscuit_ico4 | yes | 0.00503 | 0.0100 | 8 / 7,686 | 6.4e-6 |
| sphere_20480 (L5) | biscuit | yes | 0.00502 | 0.0100 | 10 / 30,726 | 6.7e-6 |
| sphere_81920 (L6) | biscuit | yes | 0.00502 | 0.0100 | 57 / 122,886 | 6.7e-6 |
| sphere_327680 (L7) | biscuit | yes | 0.00502 | 0.0100 | 277 / 491,526 | 6.7e-6 |
| icosphere_5.vtk (parc, R=1) | ugscnn | yes | 5.0e-7 | n/a | n/a | n/a |
| icosphere_6.vtk (parc, R=1) | ugscnn | yes | 5.0e-7 | n/a | n/a | n/a |

* Order matches vertex for vertex (no permutation needed) for every file; only the
  shipped 4-decimal quantisation makes |dxyz| <= 1e-6 unattainable on BISCUIT files
  (see port note). parc meshes pass the 1e-6 criterion.
* The BISCUIT spheres come in two families: `sphere_5120` (hierarchical face order, base
  ring rotated by +72 deg) and `sphere_{20480,81920,327680}` (two-pass face order). UGSCNN is a third.
* Template: all 12 `reg/*/?h.{white,inflated}.vtk` share `sphere_327680.vtk`'s face array
  exactly -> the template sphere is `sphere_327680.vtk` = `icosphere(7)`.
* Timing: level 7 builds in about 0.05 s (biscuit) / 0.1 s (ugscnn).
