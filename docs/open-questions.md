# Open questions for the owner

Answers are recorded here with date; plan rows that depend on a question
cite its ID.

| ID | Question | Why it matters | Proposed default | Answer |
|---|---|---|---|---|
| Q1 | Can we get a Linux x86-64 host with Apptainer (ACCRE or VM) to run `biscuit_FC_v2.2.sif` and keep intermediates? | Only source of acceptance references; the MATLAB orchestrator is encrypted. | Use ACCRE with a FreeSurfer `license.txt`. | |
| Q2 | Do existing BISCUIT outputs (with MaCRUISE inputs) exist on ACCRE/nfs (e.g. GF_project, ABVIB, ADSP)? | Cohort parity without recomputation; ≥10 subjects. | Collect 10–20 subjects. | |
| Q3 | Fidelity of the FreeSurfer-replaced steps: strict mimicry of `mris_make_surfaces`/`mris_inflate`/`mris_sphere`, or best Python method judged by downstream agreement? | Largest design fork. Pial from PyMaCRUISE's own outer level set is better grounded than re-deriving it from T1. | R-class replacements; pial by deforming white to PyMaCRUISE outer level set. | |
| Q4 | Hemisphere split without MNI surfaces: use SEG_refine left/right labels, add an ANTsPy affine to a bundled MNI template, or ask PyMaCRUISE to add MNI output? | BISCUIT's `mc_surf_mni` input does not exist on the PyMaCRUISE route. In the legacy sample the FreeView (scanner RAS) and MNI inner surfaces are the same mesh (250,874 vertices, identical faces) related by the rigid transform in `Surfaces_MNI/aff.txt`, so BISCUIT likely uses the MNI copy only for orientation (e.g. a midline at x = 0). Any rigid alignment to MNI would serve; the MATLAB source (requested from Lyu 2026-10-20) should confirm. | SEG_refine labels + midline plane fit; no MNI dependency. | |
| Q5 | Scope: MaCRUISE route only, or also FreeSurfer-input and CIVET routes? | FS route (input reading only, via NiBabel) isolates exact-class validation from surface-generation differences. | MaCRUISE + FS-input route; CIVET deferred. | |
| Q6 | Which templates and protocols must ship? `oasis45` (MASI production), `mb101` (image default), `hcp1058`; `braincolor`, `dkt31`. | Asset size (~1.2 GB) and data-use terms (HCP). | All, downloaded on demand. | |
| Q7 | May we contact Ilwoo Lyu for unencrypted MATLAB sources, MeshProperty/MeshContour/MeshDeform sources, and permission to redistribute models/templates? | Would replace black-box reconstruction with a direct port and settle licensing. | Yes — highest-leverage action. | 2026-10-20: owner asked Lyu for the MATLAB source; awaiting reply. Closed tools and asset permissions not yet asked. |
| Q8 | PyPI publication (public) or private index / git install only? Package name `pybiscuit` on PyPI? | Name reservation, asset hosting. | Private git install until validated; reserve name. | |
| Q9 | Required outputs for downstream users: exact legacy filenames/PDF layout (DAX processors, XNAT), or new BIDS-derivative layout only? | Determines `legacy/` view and PDF page geometry. | Both: native layout + `legacy/` mirror. | |
| Q10 | GPU support for HSD/parcellation (original has CUDA HSD)? | Runtime: HSD on CPU single-thread is slow. | CPU (Numba parallel) first; optional torch backend later. | |
| Q11 | Should pyBiscuit depend on PyMaCRUISE as a library (`pybiscuit[macruise]`, one `run` from T1 to measures) or only read its output folder? | Integration depth, version coupling. | Read outputs; optional extra provides `pybiscuit run --from-t1`. | |
| Q12 | Approve the proposed tolerances in validation-tolerances.md? | Gates every row. | — | |
