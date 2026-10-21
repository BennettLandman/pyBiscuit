"""``pybiscuit view`` (desktop window) and ``pybiscuit render`` (PNG panels).

Inputs are finished products only: ``*.surf.gii`` (or VTK) surfaces, ``*.func.gii`` /
per-vertex ``*.txt`` measures, and ``*.label.gii`` labels (LUT embedded) or integer ``*.txt``
labels with ``--parc DIR`` (``label.csv`` + ``lut.csv``). Hemisphere comes from the file name
prefix (``lh.``/``rh.``) or ``--hemi``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pybiscuit.io import (
    read_gifti,
    read_label_gii,
    read_protocol_table,
    read_vertex_txt,
    read_vtk,
)
from pybiscuit.io.labels import LabelTable
from pybiscuit.mesh.model import Mesh

from .cameras import VIEWS

_SURF_ORDER = ("white", "pial", "mid", "inflated", "sphere", "sphere.reg")


@dataclass
class Scene:
    """Everything loaded for one hemisphere."""

    hemi: str
    surfaces: dict[str, Mesh] = field(default_factory=dict)
    metrics: dict[str, np.ndarray] = field(default_factory=dict)
    labels: dict[str, tuple[np.ndarray, LabelTable]] = field(default_factory=dict)

    def surface_names(self) -> list[str]:
        key = lambda n: (_SURF_ORDER.index(n) if n in _SURF_ORDER else 99, n)  # noqa: E731
        return sorted(self.surfaces, key=key)

    def overlays(self) -> list[tuple[str, str]]:
        """``[(kind, name)]`` with kind ``metric``/``label``; first entry is always none."""
        return (
            [("none", "none")]
            + [("metric", n) for n in self.metrics]
            + [("label", n) for n in self.labels]
        )


def _hemi_of(path: Path, default: str | None) -> str:
    head = path.name.split(".")[0].split("_")[0]
    if head in ("lh", "rh"):
        return head
    if default:
        return default
    raise SystemExit(f"{path.name}: cannot infer hemisphere; use --hemi lh|rh")


def _stem(path: Path, hemi: str) -> str:
    """``lh.white.surf.gii`` -> ``white``; ``lh.curv.mean.txt`` -> ``curv.mean``."""
    parts = path.name.split(".")
    if parts[0] == hemi:
        parts = parts[1:]
    drop = {"surf", "func", "label", "gii", "txt", "vtk"}
    while len(parts) > 1 and parts[-1] in drop:
        parts.pop()
    return ".".join(parts) or path.stem


def load_scenes(files, hemi: str | None = None, parc: str | None = None) -> dict[str, Scene]:
    table = read_protocol_table(parc) if parc else None
    scenes: dict[str, Scene] = {}
    txt: list[tuple[Path, str]] = []
    for f in map(Path, files):
        if not f.exists():
            raise SystemExit(f"{f}: no such file")
        h = _hemi_of(f, hemi)
        sc = scenes.setdefault(h, Scene(h))
        name = f.name
        if name.endswith(".surf.gii") or name.endswith(".vtk"):
            m = read_gifti(f) if name.endswith(".gii") else read_vtk(f)
            sc.surfaces[_stem(f, h)] = m
            for k, v in m.point_data.items():  # VTK scalars become measures
                if np.asarray(v).ndim == 1:
                    sc.metrics.setdefault(k, np.asarray(v, float))
        elif name.endswith(".func.gii"):
            import nibabel as nib

            for i, da in enumerate(nib.load(str(f)).darrays):
                col = (da.meta or {}).get("Name") or f"{_stem(f, h)}{i or ''}"
                sc.metrics[col] = np.asarray(da.data, float)
        elif name.endswith(".label.gii"):
            sc.labels[_stem(f, h)] = read_label_gii(f)
        elif name.endswith(".txt"):
            txt.append((f, h))
        else:
            raise SystemExit(f"{name}: unsupported file type")
    for f, h in txt:
        vals = read_vertex_txt(f)
        nm = _stem(f, h)
        if table is not None and np.all(vals == np.round(vals)) and vals.min() >= 0:
            scenes[h].labels[nm] = (vals.astype(np.int32), table)
        else:
            scenes[h].metrics[nm] = vals
    for sc in scenes.values():
        if not sc.surfaces:
            raise SystemExit(f"{sc.hemi}: no surface given (*.surf.gii or *.vtk)")
        n = {m.n_vertices for m in sc.surfaces.values()}
        for k, v in list(sc.metrics.items()) + [(k, v[0]) for k, v in sc.labels.items()]:
            if len(v) not in n:
                raise SystemExit(f"{sc.hemi}.{k}: {len(v)} values do not match surface ({n})")
    return scenes


def _underlay(sc: Scene):
    for key in ("curv", "curv.mean", "sulc"):
        if key in sc.metrics:
            return sc.metrics[key]
    return None


def _overlay_kwargs(sc: Scene, kind: str, name: str) -> dict:
    if kind == "metric":
        return {"scalars": sc.metrics[name], "measure": name}
    if kind == "label":
        lab, table = sc.labels[name]
        return {"labels": lab, "table": table, "underlay": _underlay(sc)}
    return {"underlay": _underlay(sc)}


def render_main(argv) -> int:
    p = argparse.ArgumentParser(
        prog="pybiscuit render",
        description="Render PNG panels (surfaces x views) from finished products. "
        "Needs pybiscuit[viz].",
    )
    p.add_argument("files", nargs="+", help="*.surf.gii/*.vtk, *.func.gii/*.txt, *.label.gii")
    p.add_argument("--out", required=True, help="output directory for PNG panels")
    p.add_argument("--views", default="lateral,medial", help=f"comma list of {','.join(VIEWS)}")
    p.add_argument("--hemi", choices=("lh", "rh"), help="hemisphere if not in file names")
    p.add_argument("--parc", help="parcellation dir with label.csv + lut.csv (for *.txt labels)")
    p.add_argument("--measure", action="append", help="render only these overlays (repeatable)")
    p.add_argument("--tile", type=int, default=360, help="tile size in pixels (default 360)")
    p.add_argument("--html", help="also write the offline NiiVue viewer to this path")
    a = p.parse_args(argv)
    views = [v for v in a.views.split(",") if v]
    bad = set(views) - set(VIEWS)
    if bad:
        p.error(f"unknown views {sorted(bad)}")
    from . import render as R

    if not R.gl_available():
        print(
            "pybiscuit render: no off-screen OpenGL context available "
            "(on Linux try xvfb-run or an OSMesa-enabled VTK)",
            file=sys.stderr,
        )
        return 2
    scenes = load_scenes(a.files, a.hemi, a.parc)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    wrote = []
    for h, sc in sorted(scenes.items()):
        surfaces = [(n, sc.surfaces[n]) for n in sc.surface_names()]
        for kind, name in sc.overlays():
            if a.measure and name not in a.measure and kind != "none":
                continue
            if kind == "none" and (len(sc.overlays()) > 1 and a.measure):
                continue
            kw = _overlay_kwargs(sc, kind, name)
            dest = out / f"{h}.{'surfaces' if kind == 'none' else name}.png"
            R.render_panel(
                surfaces,
                h,
                views,
                tile_size=(a.tile, a.tile),
                common_scale=True,
                title=f"{h} {name}" if kind != "none" else h,
                path=dest,
                **kw,
            )
            wrote.append(dest)
    if a.html:
        from .html import collect_inputs, write_viewer

        gii = [f for f in a.files if str(f).endswith(".gii")]
        write_viewer(a.html, collect_inputs(gii))
        wrote.append(Path(a.html))
    for w in wrote:
        print(w)
    return 0


# --- interactive window ------------------------------------------------------------------


@dataclass
class ViewState:
    """Selector state for the desktop window (separate from VTK so it is testable)."""

    scene: Scene
    surface: int = 0
    overlay: int = 0

    def cycle_surface(self, step: int = 1) -> str:
        self.surface = (self.surface + step) % len(self.scene.surface_names())
        return self.scene.surface_names()[self.surface]

    def cycle_overlay(self, step: int = 1) -> tuple[str, str]:
        ov = self.scene.overlays()
        self.overlay = (self.overlay + step) % len(ov)
        return ov[self.overlay]

    def current(self) -> tuple[str, tuple[str, str]]:
        return self.scene.surface_names()[self.surface], self.scene.overlays()[self.overlay]

    def colors(self) -> tuple[np.ndarray, object]:
        from .render import vertex_colors

        sname, (kind, name) = self.current()
        return vertex_colors(self.scene.surfaces[sname], **_vc_args(self.scene, kind, name))


def _vc_args(sc: Scene, kind: str, name: str) -> dict:
    kw = _overlay_kwargs(sc, kind, name)
    return {
        k: v for k, v in kw.items() if k in ("scalars", "measure", "labels", "table", "underlay")
    }


def view_main(argv) -> int:
    p = argparse.ArgumentParser(
        prog="pybiscuit view",
        description="Desktop window (PyVista). Keys: s/S next/prev surface, m/M next/prev "
        "overlay (measure or labels), 1-6 camera presets, q quit. Needs pybiscuit[viz].",
    )
    p.add_argument("files", nargs="+")
    p.add_argument("--hemi", choices=("lh", "rh"))
    p.add_argument("--parc")
    p.add_argument("--screenshot", help="render the first state off-screen to this PNG and exit")
    a = p.parse_args(argv)
    scenes = load_scenes(a.files, a.hemi, a.parc)
    h = sorted(scenes)[0]
    state = ViewState(scenes[h])
    import pyvista as pv

    from .cameras import camera_for
    from .render import mesh_bounds, to_polydata

    pl = pv.Plotter(off_screen=bool(a.screenshot), window_size=(900, 700), title=f"pybiscuit {h}")
    pl.set_background("white")
    actor = {"a": None}
    text = {"t": None}

    def draw(reset_camera: bool = False):
        sname, (kind, name) = state.current()
        mesh = state.scene.surfaces[sname]
        rgb, style = state.colors()
        poly = to_polydata(mesh)
        poly["rgb"] = (rgb * 255).round().astype(np.uint8)
        if actor["a"] is not None:
            pl.remove_actor(actor["a"], render=False)
        actor["a"] = pl.add_mesh(
            poly, scalars="rgb", rgb=True, smooth_shading=True, show_scalar_bar=False, name="surf"
        )
        label = f"{h} {sname} | overlay: {name}"
        if style is not None:
            label += f" [{style.vmin:g}, {style.vmax:g}]"
        if text["t"] is not None:
            pl.remove_actor(text["t"], render=False)
        text["t"] = pl.add_text(label, font_size=10, color="black", position="upper_left")
        if reset_camera:
            preset("lateral")
        pl.render()

    def preset(view: str):
        sname, _ = state.current()
        pos, focal, up, scale = camera_for(view, h, mesh_bounds(state.scene.surfaces[sname]))
        pl.enable_parallel_projection()
        pl.camera.position, pl.camera.focal_point, pl.camera.up = (
            tuple(pos),
            tuple(focal),
            tuple(up),
        )
        pl.camera.parallel_scale = scale
        pl.render()

    pl.add_key_event("s", lambda: (state.cycle_surface(1), draw(True)))
    pl.add_key_event("S", lambda: (state.cycle_surface(-1), draw(True)))
    pl.add_key_event("m", lambda: (state.cycle_overlay(1), draw()))
    pl.add_key_event("M", lambda: (state.cycle_overlay(-1), draw()))
    for i, v in enumerate(VIEWS, 1):
        pl.add_key_event(str(i), lambda v=v: preset(v))
    draw(True)
    if a.screenshot:
        pl.screenshot(a.screenshot)
        pl.close()
        return 0
    pl.show()
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("view", "render"):
        print("usage: pybiscuit {view,render} ...", file=sys.stderr)
        return 2
    try:
        import matplotlib  # noqa: F401
        import pyvista  # noqa: F401
    except ImportError:
        print(
            f"pybiscuit {argv[0]} needs the viz extra: pip install 'pybiscuit[viz]'",
            file=sys.stderr,
        )
        return 2
    return (view_main if argv[0] == "view" else render_main)(argv[1:])
