# Licensing inventory

pyBiscuit's own code: **Apache-2.0** (`LICENSE`, `NOTICE`). Ported code keeps
upstream copyright notices in file headers and in `LICENSES/`. How restricted
code (GCO/GCMex, FreeSurfer) was avoided: [clean-room.md](clean-room.md).

## Code

| Component | License | Port allowed? | Action |
|---|---|---|---|
| BISCUIT (`/opt/biscuit/LICENSE`, © 2019 Ilwoo Lyu) | MIT | yes | attribute |
| CMorph (`/opt/cmorph/LICENSE`, © 2018 Ilwoo Lyu) | MIT | yes | attribute |
| HSD, CurveExtraction, LocalGyrificationIndex, SphericalRemesh, SphericalLabeling, MeshLib | MIT (© Ilwoo Lyu) | yes | attribute |
| `HSD/src/bobyqa.h` | © Powell (Fortran), Thiébaut (C), Lyu (header); see its LICENSE | verify text | read accompanying license before porting |
| MeshLib geodesic (`Geodesic/gw`) | © 2004 Gabriel Peyré; covered by MeshLib MIT (confirmed 2026-11-03) | yes | attribute |
| klaplace | **no LICENSE file**; bundles SimpleOpt (MIT) | yes: authors approved Apache-2.0 (2026-11-03) | ported (`quant/klaplace.py`); credited in `NOTICE` |
| SurfRemesh | **no LICENSE file** | unclear | not needed on MaCRUISE route |
| ShapeComplexityIndex (NIRALUser) | **no LICENSE file**; `transportSimplex.h` © Darren MacDonald, parts © Yossi Rubner | yes: authors approved Apache-2.0 (2026-11-03) | ported (`quant/sci.py`, closed-form 1D EMD; `transportSimplex.h` not ported); credited in `NOTICE` |
| GCMex (Shai Bagon) / GCO (Boykov, Veksler, Kolmogorov) | research-only terms (GCO v3) | **no** for redistribution | implement alpha-expansion on PyMaxflow (GPL? — check) or own max-flow; verify licence first |
| MeshProperty, MeshDeform, MeshContour | closed binaries (Lyu) | yes: approved by Lyu (2026-11-03) | black-box + disassembly re-implementation |
| FreeSurfer 6 binaries | FreeSurfer license (not OSI) | **no** | clean-room replacements; never read FS source |
| Mesh voxelisation `VOXELISE.m` (Aitkenhead; in LocalGyrificationIndex) | BSD-3-Clause (© 2013 Adam H. Aitkenhead) | yes | attribute; text in `LICENSES/VOXELISE.txt` |
| UGSCNN MeshConv (`meshcnn/`) | from Jiang et al. (MIT) via SphericalLabeling | yes | attribute |
| NiiVue 0.69.0 (`pybiscuit/viz/data/niivue.umd.js`, © 2021 Niivue) | BSD-2-Clause; UMD inlines MIT deps (fflate, gl-matrix, nifti-reader-js, zarrita, …) | yes | bundled in `[viz]` only; text in `LICENSES/NiiVue.txt`; version + SHA-256 in `viz/data/NIIVUE.json` |

## Models and templates

| Asset | Origin | Concern |
|---|---|---|
| `parc/braincolor/*.pth` | trained on BrainCOLOR-labelled data (Mindboggle/OASIS/NIREP?) | redistribution permission from Lyu / data owners |
| `parc/dkt31/*.pth` | trained on Mindboggle-101 DKT31 (CC-BY?) | same |
| `reg/mb101` | Mindboggle-101 (CC BY-NC-SA? verify) | derived means/variances; confirm |
| `reg/oasis45` | OASIS | OASIS data-use terms |
| `reg/hcp1058` | HCP | HCP Open Access Data Use Terms restrict redistribution of derived data; confirm |

Default plan: host assets outside PyPI (a GitHub Release,
or Zenodo/HF once cleared), download on demand with SHA-256 pinning
(`pybiscuit fetch-assets`). Owner to confirm what may be redistributed and
where.
