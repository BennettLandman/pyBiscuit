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
| UGSCNN-MeshConv.txt | UGSCNN MeshConv | (c) 2018 Chiyu "Max" Jiang, UC Berkeley (MIT) | `github.com/maxjiang93/ugscnn` LICENSE at commit `64e27969c65b4807e41c1e275972dab8bb72c512` (default-branch HEAD, fetched 2026-10-20) |

## Not included (license unclear or incompatible)

- klaplace (no LICENSE file; bundles SimpleOpt, MIT) - unclear, ask upstream.
- SurfRemesh (no LICENSE file) - not needed on the MaCRUISE route.
- ShapeComplexityIndex (NIRALUser; no LICENSE file; transportSimplex.h
  copyrights Darren MacDonald / Yossi Rubner) - unclear.
- HSD `bobyqa.h` (Powell / Thiebaut) - accompanying license to be verified.
- MeshLib geodesic (Peyre 2004) - terms to be confirmed.
- GCMex / GCO - research-only terms; not redistributable.
- MeshProperty, MeshDeform, MeshContour - closed binaries.
- FreeSurfer - non-OSI license; clean-room replacements only.
- Models/templates (parc `*.pth`, reg `mb101`/`oasis45`/`hcp1058`) - data
  terms unresolved; not in git.
