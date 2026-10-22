"""FreeSurfer-route input stage (``run_postproc_fs_pipeline.m``; spec §12).

From a FreeSurfer ``recon-all`` subject directory (``surf/``, ``label/``) produce exactly the
files BISCUIT's FreeSurfer route puts in its job directory before registration:

* ``<h>.{white,gray,inflated,sphere}.vtk`` — each surface read from ``surf/<h>.<name>`` (``pial``
  is written as ``gray``) and written by ``write_vtk.m``: ``POINTS n float`` with ``%f`` (six
  decimals), ``POLYGONS`` with 0-based faces, no point data, FreeSurfer's own (tkRAS) frame;
* ``<h>.mid.vtk`` — ``(white + gray) / 2`` of the *printed* vertices (the MATLAB script re-reads
  the VTKs it just wrote), faces of ``white``;
* ``<h>.<feature>.txt`` — ``curv``, ``sulc``, ``inflated.H``, ``thickness`` as ``%f\\n``;
* ``<h>.mask.txt`` — 1 for the vertices listed in ``label/<h>.cortex.label``, else 0 (``%d\\n``).

``surf_smooth`` > 0 (``MeshDeform --smooth``) is not ported; it is 0 in the FreeSurfer route as
the image runs it (checked against the three open examples).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import read_fs_curv, read_fs_surf, read_vtk

__all__ = [
    "FEATURES", "SURFACES", "format_f", "mask_from_label", "mid_vertices", "prepare_fs_subject",
    "read_cortex_label", "vtk_text", "write_feature_txt", "write_mask_txt", "write_matlab_vtk",
]  # fmt: skip

#: FreeSurfer ``surf/`` name -> BISCUIT name (``run_postproc_fs_pipeline.m:19-24``)
SURFACES = {"white": "white", "pial": "gray", "inflated": "inflated", "sphere": "sphere"}
#: curv files converted to ``<h>.<name>.txt``
FEATURES = ("curv", "sulc", "inflated.H", "thickness")


def format_f(values) -> str:
    """``sprintf('%f\\n', v)`` for every element of a 1-D array (six decimals)."""
    return "".join(f"{x:f}\n" for x in np.asarray(values, dtype=np.float64).tolist())


def vtk_text(vertices, faces) -> str:
    """The text ``write_vtk.m`` writes: header, ``POINTS n float`` (``%f``), ``POLYGONS``."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    n, m = v.shape[0], f.shape[0]
    rows = "".join(f"{a:f} {b:f} {c:f}\n" for a, b, c in v.tolist())
    tris = "".join(f"3 {a} {b} {c}\n" for a, b, c in f.tolist())
    return (
        "# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n"
        f"POINTS {n} float\n{rows}POLYGONS {m} {4 * m}\n{tris}"
    )


def write_matlab_vtk(path, vertices, faces) -> None:
    Path(path).write_text(vtk_text(vertices, faces))


def printed(vertices) -> np.ndarray:
    """Vertices as read back from a ``%f`` print: rounded to six decimals (float64)."""
    return np.array([[float(f"{x:f}") for x in row] for row in np.asarray(vertices).tolist()])


def mid_vertices(white, gray) -> np.ndarray:
    """``(v_wm + v_gm) / 2`` where both come from the printed VTKs (``:41-47``)."""
    return (printed(white) + printed(gray)) / 2


def read_cortex_label(path) -> np.ndarray:
    """Vertex indices (0-based) of a FreeSurfer label file: ``dlmread(path, ' ', 2, 0)(:, 1)``.

    Line 1 is a comment, line 2 the count, then ``vertex x y z value`` rows.
    """
    lines = Path(path).read_text().splitlines()
    n = int(lines[1].split()[0])
    return np.array([int(float(ln.split()[0])) for ln in lines[2 : 2 + n]], dtype=np.int64)


def mask_from_label(label_vertices, n_vertices: int) -> np.ndarray:
    mask = np.zeros(n_vertices, dtype=np.int64)
    mask[np.asarray(label_vertices, dtype=np.int64)] = 1
    return mask


def write_feature_txt(curv, path) -> None:
    """``fprintf(fp, '%f\\n', read_curv(...))``."""
    Path(path).write_text(format_f(curv))


def write_mask_txt(mask, path) -> None:
    Path(path).write_text("".join(f"{int(x)}\n" for x in mask))


def prepare_fs_subject(subject_dir, job_dir, *, hemis=("lh", "rh"), force: bool = False) -> dict:
    """Produce the FreeSurfer-route job directory contents for ``hemis``.

    ``subject_dir`` is the ``recon-all`` directory (with ``surf/`` and optionally ``label/``).
    Existing outputs are kept unless ``force`` (the MATLAB script is idempotent the same way).
    Returns ``{hemi: {name: path}}`` of what exists afterwards.
    """
    subject_dir, job_dir = Path(subject_dir), Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    made: dict[str, dict[str, Path]] = {}
    for h in hemis:
        out = made.setdefault(h, {})
        for name in FEATURES:
            dst = job_dir / f"{h}.{name}.txt"
            if force or not dst.exists():
                write_feature_txt(read_fs_curv(subject_dir / "surf" / f"{h}.{name}"), dst)
            out[name] = dst
        for src, dstname in SURFACES.items():
            dst = job_dir / f"{h}.{dstname}.vtk"
            if force or not dst.exists():
                m = read_fs_surf(subject_dir / "surf" / f"{h}.{src}")
                write_matlab_vtk(dst, m.vertices, m.faces)
            out[dstname] = dst
        mid = job_dir / f"{h}.mid.vtk"
        if force or not mid.exists():
            wm, gm = read_vtk(job_dir / f"{h}.white.vtk"), read_vtk(job_dir / f"{h}.gray.vtk")
            write_matlab_vtk(mid, mid_vertices(wm.vertices, gm.vertices), wm.faces)
        out["mid"] = mid
        label = subject_dir / "label" / f"{h}.cortex.label"
        mask = job_dir / f"{h}.mask.txt"
        if label.exists() and (force or not mask.exists()):
            n = read_vtk(job_dir / f"{h}.white.vtk").n_vertices
            write_mask_txt(mask_from_label(read_cortex_label(label), n), mask)
        if mask.exists():
            out["mask"] = mask
    return made
