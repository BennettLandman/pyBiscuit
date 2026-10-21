# Port note: visualization (`pybiscuit.viz`, `pybiscuit.io` exports)

Not a port of upstream code: new code following [visualization-plan.md](../visualization-plan.md). Sources:
colour limits from the original `biscuit.pdf` page-5 conventions listed there; GIfTI metadata names from
Connectome Workbench's published GIfTI conventions (`AnatomicalStructurePrimary`, `GeometricType`); NiiVue 0.69.0
(BSD-2) bundled unmodified. No FreeSurfer code read. Floating point: GIfTI surfaces are float32 on disk; labels
int32. Evidence: [report](../reports/viz-foundation-2026-10-21.md).
