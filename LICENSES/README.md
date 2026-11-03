# Third-party licenses

Verbatim license texts for upstream components that pyBiscuit ports or derives
from. Copied from the LICENSE files under `reference/` (extracted image and
upstream clones). See `docs/licensing.md` for the full inventory.

| File | Component | Copyright line | Source |
|---|---|---|---|
| BISCUIT.txt | BISCUIT | (c) 2019 Ilwoo Lyu | `/opt/biscuit/LICENSE` |
| CMorph.txt | CMorph | (c) 2018 Ilwoo Lyu | `/opt/cmorph/LICENSE` |
| HSD.txt | HSD | (c) 2018 Ilwoo Lyu | `repos/HSD/LICENSE` |
| CurveExtraction.txt | CurveExtraction | (c) 2018 Ilwoo Lyu | `repos/CurveExtraction/LICENSE` |
| LocalGyrificationIndex.txt | LocalGyrificationIndex | (c) 2017 Ilwoo Lyu | `repos/LocalGyrificationIndex/LICENSE` |
| SphericalRemesh.txt | SphericalRemesh | (c) 2016 Ilwoo Lyu | `repos/SphericalRemesh/LICENSE` |
| SphericalLabeling.txt | SphericalLabeling | (c) 2019 Ilwoo Lyu | `repos/SphericalLabeling/LICENSE` |
| MeshLib.txt | MeshLib | (c) 2009 Ilwoo Lyu | `repos/MeshLib/LICENSE` |
| VOXELISE.txt | Mesh voxelisation (VOXELISE.m) | (c) 2013 Adam H. Aitkenhead (BSD-3-Clause) | MATLAB File Exchange 27390 v1.20.0 `license.txt` (downloaded 2026-11-03) |
| UGSCNN-MeshConv.txt | UGSCNN MeshConv | (c) 2018 Chiyu "Max" Jiang, UC Berkeley (MIT) | `github.com/maxjiang93/ugscnn` LICENSE at commit `64e27969c65b4807e41c1e275972dab8bb72c512` (default-branch HEAD, fetched 2026-10-20) |

## Ported by permission (no upstream license file)

- klaplace (Joohwi Lee, UNC) and ShapeComplexityIndex (Sun Hyung Kim, Martin
  Styner, UNC): the authors approved release under Apache-2.0 (2026-11-03).
  `transportSimplex.h` (MacDonald / Rubner) was not ported.
- MeshProperty, MeshDeform, MeshContour (Ilwoo Lyu; closed binaries): approved
  by the author (2026-11-03).
- MeshLib geodesic (Peyre 2004): confirmed covered by MeshLib's MIT license
  (2026-11-03; `MeshLib.txt`).

## Not included (license unclear or incompatible)

- SurfRemesh (no LICENSE file) - not needed on the MaCRUISE route.
- HSD `bobyqa.h` (Powell / Thiebaut) - accompanying license to be verified.
- GCMex / GCO - research-only terms; not redistributable.
- FreeSurfer - non-OSI license; clean-room replacements only.
- Models/templates (parc `*.pth`, reg `mb101`/`oasis45`/`hcp1058`) - data
  terms unresolved; not in git.
