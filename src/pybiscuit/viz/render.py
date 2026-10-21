"""Off-screen surface renderer (PyVista) and panel composer (matplotlib).

Colours are computed here (colormap + fixed limits, or the label LUT) and handed to VTK as
per-vertex RGB, so images do not depend on VTK scalar-mapping defaults.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from pybiscuit.mesh.model import Mesh

from .cameras import camera_for
from .colormaps import MeasureStyle, measure_style

BACKGROUND = (1.0, 1.0, 1.0)
UNDERLAY_GRAY = (0.62, 0.62, 0.62)  # base grey for label/curve overlays
NAN_COLOR = (0.75, 0.75, 0.75)


@lru_cache(maxsize=1)
def gl_available() -> bool:
    """True when an off-screen OpenGL context can be created (probed in a subprocess)."""
    if os.environ.get("PYBISCUIT_NO_GL"):
        return False
    code = (
        "import pyvista as pv;p=pv.Plotter(off_screen=True,window_size=(16,16));"
        "p.add_mesh(pv.Sphere());img=p.screenshot(return_img=True);"
        "raise SystemExit(0 if img.std()>0 else 1)"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def mesh_bounds(mesh: Mesh) -> np.ndarray:
    """(xmin, xmax, ymin, ymax, zmin, zmax) of the vertices."""
    v = mesh.vertices
    return np.stack([v.min(0), v.max(0)], axis=1).ravel()


def scalar_colors(values, style: MeasureStyle) -> np.ndarray:
    """Map ``values`` to float RGB (N, 3) with the style's fixed limits; NaN -> grey."""
    import matplotlib

    v = np.asarray(values, float)
    t = np.clip((v - style.vmin) / (style.vmax - style.vmin), 0, 1)
    rgb = np.asarray(matplotlib.colormaps[style.cmap](np.nan_to_num(t))[:, :3])
    rgb[~np.isfinite(v)] = NAN_COLOR
    return rgb


def label_colors(labels, table, background=UNDERLAY_GRAY) -> np.ndarray:
    """LUT colours per vertex (N, 3); labels not in ``table`` (e.g. 0) get ``background``."""
    lab = np.asarray(labels).astype(np.int64)
    rgb = np.tile(np.asarray(background, float), (lab.size, 1))
    for i in np.unique(lab):
        if int(i) in table.entries:
            rgb[lab == i] = np.asarray(table.rgb(int(i)), float) / 255
    return rgb


def shade_colors(values, threshold=0.0, dark=0.35, light=0.8) -> np.ndarray:
    """Two-tone grey underlay from a signed map (e.g. curvature): negative dark, positive light."""
    g = np.where(np.asarray(values, float) > threshold, light, dark)
    return np.repeat(g[:, None], 3, axis=1)


def to_polydata(mesh: Mesh):
    import pyvista as pv

    faces = np.hstack([np.full((mesh.n_faces, 1), 3, np.int64), mesh.faces.astype(np.int64)])
    return pv.PolyData(mesh.vertices.copy(), faces.ravel())


def vertex_colors(
    mesh: Mesh,
    scalars=None,
    measure: str | None = None,
    style: MeasureStyle | None = None,
    labels=None,
    table=None,
    underlay=None,
) -> tuple[np.ndarray, MeasureStyle | None]:
    """Resolve the per-vertex RGB for the requested overlay; returns ``(rgb, style_used)``."""
    if labels is not None:
        if table is None:
            raise ValueError("labels need a LabelTable")
        lab = np.asarray(labels)
        rgb = label_colors(lab, table)
        if underlay is not None:
            unl = ~np.isin(lab, list(table.entries))
            rgb[unl] = shade_colors(underlay)[unl]
        return rgb, None
    if scalars is None and measure is not None:
        scalars = mesh.point_data[measure]
    if scalars is not None:
        if style is None:
            style = measure_style(measure or "data", scalars)
        return scalar_colors(scalars, style), style
    if underlay is not None:
        return shade_colors(underlay), None
    return np.tile(np.asarray(UNDERLAY_GRAY), (mesh.n_vertices, 1)), None


def render_surface(
    mesh: Mesh,
    hemi: str = "lh",
    view: str = "lateral",
    *,
    scalars=None,
    measure: str | None = None,
    style: MeasureStyle | None = None,
    labels=None,
    table=None,
    underlay=None,
    curves: Sequence[np.ndarray] = (),
    curve_color=(0.9, 0.1, 0.1),
    curve_radius: float = 0.4,
    size: tuple[int, int] = (480, 480),
    bounds=None,
    background=BACKGROUND,
) -> np.ndarray:
    """Render one surface from one preset view; returns an RGB uint8 image (H, W, 3).

    ``scalars``/``measure``: colour by a scalar with the measure's fixed limits (``measure``
    may also name a ``mesh.point_data`` entry). ``labels``+``table``: colour by LUT.
    ``underlay``: signed map (e.g. curvature) giving a sulcal/gyral grey base.
    ``curves``: polylines (K, 3) drawn as tubes. ``bounds`` fixes the field of view
    (pass the same bounds to several surfaces to render them at a common scale).
    """
    import pyvista as pv

    rgb, _ = vertex_colors(mesh, scalars, measure, style, labels, table, underlay)
    poly = to_polydata(mesh)
    poly["rgb"] = (rgb * 255).round().astype(np.uint8)
    pl = pv.Plotter(off_screen=True, window_size=list(size))
    pl.set_background(background)
    pl.add_mesh(
        poly,
        scalars="rgb",
        rgb=True,
        smooth_shading=True,
        show_scalar_bar=False,
        specular=0.1,
        ambient=0.35,
        diffuse=0.75,
    )
    for pts in curves:
        pts = np.asarray(pts, float)
        if len(pts) < 2:
            continue
        line = pv.lines_from_points(pts)
        pl.add_mesh(line.tube(radius=curve_radius), color=curve_color, show_scalar_bar=False)
    pos, focal, up, scale = camera_for(view, hemi, mesh_bounds(mesh) if bounds is None else bounds)
    pl.enable_parallel_projection()
    pl.camera.position = tuple(pos)
    pl.camera.focal_point = tuple(focal)
    pl.camera.up = tuple(up)
    pl.camera.parallel_scale = scale
    pl.camera.clipping_range = (0.01, float(np.linalg.norm(pos - focal)) * 3)
    img = pl.screenshot(return_img=True)
    pl.close()
    return np.asarray(img)


def compose_panel(
    tiles: Sequence[Sequence[np.ndarray]],
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    style: MeasureStyle | None = None,
    title: str | None = None,
    path=None,
    dpi: int = 100,
) -> np.ndarray:
    """Lay out ``tiles[row][col]`` images (N surfaces x M views) with labels and a colourbar.

    Writes a PNG when ``path`` is given; returns the RGB uint8 canvas either way.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    nr, nc = len(tiles), len(tiles[0])
    h, w = tiles[0][0].shape[:2]
    top = 0.6 if (title and col_labels) else (0.35 if (title or col_labels) else 0.0)
    bottom = 0.6 if style is not None else 0.0
    left = 1.1 if row_labels else 0.0
    fig = Figure(figsize=(left + nc * w / dpi, top + bottom + nr * h / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    fw, fh = fig.get_size_inches()
    for r in range(nr):
        for c in range(nc):
            rect = [
                (left + c * w / dpi) / fw,
                (bottom + (nr - 1 - r) * h / dpi) / fh,
                (w / dpi) / fw,
                (h / dpi) / fh,
            ]
            ax = fig.add_axes(rect)
            ax.imshow(tiles[r][c])
            ax.axis("off")
            if r == 0 and col_labels:
                ax.set_title(col_labels[c], fontsize=10)
            if c == 0 and row_labels:
                ax.text(-0.02, 0.5, row_labels[r], transform=ax.transAxes, ha="right", va="center")
    if title:
        fig.suptitle(title, fontsize=11, y=0.995, va="top")
    if style is not None:
        span = 0.6 * nc * w / dpi
        cax = fig.add_axes([(left + 0.2 * nc * w / dpi) / fw, 0.25 / fh, span / fw, 0.12 / fh])
        sm = ScalarMappable(Normalize(style.vmin, style.vmax), cmap=style.cmap)
        fig.colorbar(sm, cax=cax, orientation="horizontal", label=style.label)
    fig.canvas.draw()
    canvas = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
    return canvas


def render_panel(
    surfaces: Sequence[tuple[str, Mesh]],
    hemi: str,
    views: Sequence[str] = ("lateral", "medial"),
    *,
    common_scale: bool = False,
    tile_size: tuple[int, int] = (360, 360),
    show_colorbar: bool = True,
    title: str | None = None,
    path=None,
    **overlay,
) -> np.ndarray:
    """N surfaces x M views -> one PNG. ``surfaces`` is ``[(name, mesh), ...]``.

    Overlay kwargs (``measure``, ``scalars``, ``labels``, ``table``, ``underlay``, ``curves``,
    ...) go to :func:`render_surface`; a dict value keyed by surface name selects per-surface data.
    """
    style = overlay.get("style")
    if style is None and (overlay.get("measure") or overlay.get("scalars") is not None):
        data = overlay.get("scalars")
        if isinstance(data, dict):
            data = next(iter(data.values()))
        style = measure_style(overlay.get("measure") or "data", data)
        overlay["style"] = style
    bounds = None
    if common_scale:
        allb = np.array([mesh_bounds(m) for _, m in surfaces]).reshape(-1, 3, 2)
        bounds = np.stack([allb[:, :, 0].min(0), allb[:, :, 1].max(0)], axis=1).ravel()
    tiles = []
    for name, mesh in surfaces:
        kw = {k: (v[name] if isinstance(v, dict) and name in v else v) for k, v in overlay.items()}
        tiles.append(
            [render_surface(mesh, hemi, v, size=tile_size, bounds=bounds, **kw) for v in views]
        )
    colorbar_style = style if show_colorbar and overlay.get("labels") is None else None
    return compose_panel(
        tiles,
        row_labels=[n for n, _ in surfaces],
        col_labels=list(views),
        style=colorbar_style,
        title=title,
        path=path,
    )
