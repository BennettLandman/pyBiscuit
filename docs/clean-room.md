# Code provenance and clean-room compliance

pyBiscuit is distributed under Apache-2.0 (`LICENSE`, `NOTICE`). Every module
is either (a) a port of upstream code whose license permits it (MIT or BSD,
texts in `LICENSES/`), or (b) written from scratch from published papers
and the observed behavior of the reference image, without reading or
translating code under incompatible terms. This page records the rules the
project followed and how each restricted component was handled. The
per-module port notes in `docs/ports/` give file and line-level detail.

## Rules

These rules were written into the agent guide ([AGENTS.md](../AGENTS.md),
"Working rules") before the porting started, and every contributor had to
follow them:

1. **FreeSurfer clean-room rule.** Do not read or translate FreeSurfer
   source code. Replacements are written from papers and black-box behavior
   only, because FreeSurfer's license is not compatible.
2. **Dependency rule.** Runtime dependencies must be MIT, BSD, Apache or PSF
   licensed. No GPL or LGPL code in the default install.
3. **Third-party code inside BISCUIT.** The BISCUIT MATLAB tree bundles code
   that is not MIT: FreeSurfer-licensed `read_*`/`write_*` helpers and the
   GCMex/GCoptimization/maxflow graph cut. These files must not be copied
   ([orchestrator-spec](reference/orchestrator-spec.md), "Licence" and §14).

## Graph cut (parcellation labels): written from scratch

BISCUIT labels the parcellation with Shai Bagon's GCMex, which wraps Olga
Veksler's GCoptimization and the Boykov–Kolmogorov `maxflow` library. These
are licensed GPL v2 with an extra research-only clause and a patent notice,
so they can be neither redistributed nor translated.

- **Decision:** the owner ruled on 2026-10-21
  ([open-questions](open-questions.md) Q13) that the graph cut be written
  independently from the published algorithms. No GCO, GCMex or maxflow code
  was to be read or translated, and tie-level label differences are
  accepted.
- **What was done:** `src/pybiscuit/parc/graphcut.py` implements
  α-expansion from Boykov, Veksler & Zabih (2001), *Fast Approximate Energy
  Minimization via Graph Cuts*, IEEE TPAMI 23(11):1222–1239. The binary
  graph construction follows Kolmogorov & Zabih (2004), *What Energy
  Functions Can Be Minimized via Graph Cuts?*, IEEE TPAMI 26(2):147–159. The
  min-cut uses a Dinic blocking-flow solver written for this package. It is
  not the Boykov–Kolmogorov algorithm that `maxflow` uses.
- **What was ported:** only BISCUIT's own MIT-licensed MATLAB callers
  (`gcut.m`, `gcut_.m`, `roi_correction.m`). They define the energy: the
  data cost, the Potts weights, the label order and the stopping rule. No
  file in `SurfPostproc/common/matlab/gcut/` was opened.
- **Records:** module docstring, [ports/graphcut.md](ports/graphcut.md)
  ("Licence & clean-room implementation"), and the report
  [graphcut-cleanup-2026-10-22](reports/graphcut-cleanup-2026-10-22.md).

## FreeSurfer: clean-room replacements

- **File formats:** FreeSurfer surfaces, curvature files, annotations and
  labels are read with NiBabel (MIT). BISCUIT's FreeSurfer-licensed
  `read_surf.m` and `read_curv.m` were not read
  ([ports/freesurfer-route.md](ports/freesurfer-route.md)).
- **Tools:** the `mri_*`/`mris_*` steps BISCUIT calls are rebuilt in
  `src/pybiscuit/surfprep/` from three sources: BISCUIT's own MIT
  orchestration code, papers, and the inputs and outputs of the reference
  image. They are judged by whether the downstream measures
  agree, not by matching FreeSurfer's code. No FreeSurfer source is in this
  repository or in the local reference tree it was built from.
- **Image runs:** the reference image runs its own FreeSurfer binaries under
  the user's FreeSurfer license. pyBiscuit neither ships nor calls them.

## Other restricted components

| Component | Terms | Status in pyBiscuit |
|---|---|---|
| HSD `bobyqa.h` (Powell, Thiébaut) | separate license, not verified | **Not ported.** Only the landmark pole search uses it, and the CLI never reaches that code ([ports/hsd.md](ports/hsd.md)). |
| klaplace (Joohwi Lee, UNC) | no license file | **Ported from source; permission requested.** Do not publish `quant/klaplace.py` until the authors grant permission. |
| ShapeComplexityIndex (Sun Hyung Kim, Martin Styner, UNC) | "Copyright (C) 2015", no license | **Ported from source; permission requested.** Same hold for `quant/sci.py`. The EMD is a closed-form 1D version; `transportSimplex.h` (MacDonald/Rubner) was not ported. |
| MeshProperty, MeshDeform, MeshContour (Ilwoo Lyu) | closed binaries, no license | Ported from the binaries' observed behavior plus disassembly ([ports/meshproperty.md](ports/meshproperty.md)). They are thin wrappers over MeshLib (MIT). **Confirm with Ilwoo Lyu** (Q7). |
| MeshLib geodesic `gw` (Gabriel Peyré, 2004) | inside the MIT MeshLib repo; the files carry no license header | Ported in `quant/_fastmarch.py` and `mesh/geodesic.py`. Treated as MeshLib MIT; confirm. |
| `VOXELISE.m` (Adam Aitkenhead) | inside the MIT LocalGyrificationIndex repo; MATLAB File Exchange code | Ported in `quant/_voxelise.py`. Confirm the File Exchange (BSD) terms and add the text to `LICENSES/`. |
| SurfRemesh | no license file | Not ported (not used on the MaCRUISE route). |
| Parcellation models, registration templates | dataset terms (HCP, OASIS, Mindboggle) | Not in git. Fetched on demand once redistribution is cleared ([licensing.md](licensing.md)). |

Rows marked **permission requested** or **confirm** must be resolved before
the repository is made public.
