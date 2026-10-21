"""Render the B12 example gallery from template and PyMaCRUISE products (needs [viz], GL).

Usage: python scripts/viz_gallery.py OUT_DIR
Inputs live outside git (reference/ and ~/dev/mipav); missing inputs are skipped with a note.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pybiscuit.io import read_vertex_txt, read_vtk
from pybiscuit.viz.render import render_panel

REPO = Path(__file__).resolve().parents[1]
OPT = REPO.parent / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit"
MIPAV = Path.home() / "dev" / "mipav" / "pymacruise" / "build"
INNER = MIPAV / "ace-indexed-heap-2026-10-08" / "base-surface-thickness" / "inner_surface_thickness.vtk"


def _field_array(path: Path, name: str):
    """Read one ``FIELD PointData`` array (``read_vtk`` does not parse FIELD blocks yet)."""
    import numpy as np

    lines = path.read_text().splitlines()
    for i, ln in enumerate(lines):
        parts = ln.split()
        if parts[:1] == [name] and len(parts) == 4:
            return np.array(lines[i + 1 : i + 1 + int(parts[2])], dtype=float)
    raise KeyError(name)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    reg = OPT / "reg" / "mb101"
    if reg.exists():
        for hemi in ("lh", "rh"):
            surfs = [(k, read_vtk(reg / f"{hemi}.{k}.vtk")) for k in ("white", "inflated")]
            for feat in ("curv", "sulc"):
                vals = read_vertex_txt(reg / f"{hemi}.{feat}.mean.txt")
                render_panel(
                    surfs, hemi, ["lateral", "medial", "dorsal"], measure=feat, scalars=vals,
                    title=f"mb101 template {hemi} {feat} mean", path=out / f"template_{hemi}_{feat}.png",
                    common_scale=True,
                )
        sph = read_vtk(OPT / "sphere" / "sphere_20480.vtk")
        curv = read_vertex_txt(reg / "lh.curv.mean.txt")
        from pybiscuit.mesh.model import Mesh

        # sphere_20480 is a coarser icosphere than the template: show a smooth analytic scalar
        render_panel(
            [("sphere_20480", Mesh(sph.vertices * 100 / abs(sph.vertices).max(), sph.faces))],
            "lh", ["lateral", "medial", "dorsal"], measure="curv",
            scalars=(sph.vertices[:, 2] / abs(sph.vertices[:, 2]).max()) * 0.4,
            title="shipped sphere_20480 (z coordinate as scalar)", path=out / "sphere_20480.png",
        )
        del curv
    else:
        print("template assets missing; skipped template and sphere panels")
    if INNER.exists():
        m = read_vtk(INNER)
        thick = _field_array(INNER, "mean_thickness_mm")
        render_panel(
            [("inner surface", m)], "lh", ["lateral", "dorsal", "anterior"], measure="thickness",
            scalars=thick, title="PyMaCRUISE inner surface, thickness (1-5 mm)",
            path=out / "pymacruise_inner_thickness.png", tile_size=(420, 420),
        )
    else:
        print("PyMaCRUISE inner surface missing; skipped")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
