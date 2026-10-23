"""Self-contained offline viewer: ``qa/index.html`` built on a bundled NiiVue.

The page inlines the pinned NiiVue UMD build (``data/niivue.umd.js``, see ``data/NIIVUE.json``
for version, licence and SHA-256) and, by default, the GIfTI/NIfTI files themselves as base64,
so it opens from ``file://`` with no network and no web server. ``embed=False`` writes
relative file references instead (smaller; needs an HTTP server because browsers block
``fetch`` from ``file://``).
"""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib

from .colormaps import measure_style

DATA_DIR = Path(__file__).parent / "data"


def niivue_info() -> dict:
    """Pinned NiiVue provenance (version, licence, SHA-256 of the bundled file)."""
    return json.loads((DATA_DIR / "NIIVUE.json").read_text())


def _niivue_js() -> str:
    return (
        (DATA_DIR / "niivue.umd.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
    )


# matplotlib maps from colormaps.py: NiiVue has "viridis" built in; others are sampled and
# registered in the page with ``nv.addColormap`` so colours match the PNG renderer.
_NV_BUILTIN = {"viridis"}


def _sampled_colormap(name: str, n: int = 32) -> dict:
    try:
        import matplotlib

        cm = matplotlib.colormaps[name]
        rgba = [cm(i / (n - 1)) for i in range(n)]
        return {
            "R": [round(c[0] * 255) for c in rgba],
            "G": [round(c[1] * 255) for c in rgba],
            "B": [round(c[2] * 255) for c in rgba],
            "A": [255] * n,
            "I": [round(i * 255 / (n - 1)) for i in range(n)],
        }
    except (ImportError, KeyError):
        if name in ("RdBu_r", "RdBu"):
            # Diverging red-blue (ColorBrewer 11-class reversed)
            nodes_hex = [
                "#053061",
                "#2166ac",
                "#4393c3",
                "#92c5de",
                "#d1e5f0",
                "#f7f7f7",
                "#fddbc7",
                "#f4a582",
                "#d6604d",
                "#b2182b",
                "#67001f",
            ]
            if name == "RdBu":
                nodes_hex = nodes_hex[::-1]
            nodes_rgb = [(int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)) for h in nodes_hex]
            import numpy as np

            x_nodes = np.linspace(0, 1, len(nodes_hex))
            x_eval = np.linspace(0, 1, n)
            r = np.interp(x_eval, x_nodes, [c[0] for c in nodes_rgb])
            g = np.interp(x_eval, x_nodes, [c[1] for c in nodes_rgb])
            b = np.interp(x_eval, x_nodes, [c[2] for c in nodes_rgb])
            return {
                "R": [int(round(v)) for v in r],
                "G": [int(round(v)) for v in g],
                "B": [int(round(v)) for v in b],
                "A": [255] * n,
                "I": [round(i * 255 / (n - 1)) for i in range(n)],
            }
        raise


@dataclass
class ViewerInputs:
    """Files shown by the viewer. All paths are GIfTI (``volume``: NIfTI)."""

    surfaces: list[tuple[str, str, Path]] = field(default_factory=list)  # (hemi, kind, path)
    metrics: list[tuple[str, str, Path]] = field(default_factory=list)  # (hemi, measure, path)
    labels: list[tuple[str, str, Path]] = field(default_factory=list)  # (hemi, name, path)
    volume: Path | None = None
    title: str = "pyBiscuit QA"


def _hemi_kind(name: str) -> tuple[str, str]:
    """``lh.white.surf.gii`` -> ("lh", "white")."""
    parts = name.split(".")
    if len(parts) < 3 or parts[0] not in ("lh", "rh"):
        raise ValueError(
            f"cannot infer hemisphere/kind from {name!r}; expected e.g. lh.white.surf.gii"
        )
    return parts[0], ".".join(parts[1:-2])


def collect_inputs(paths, volume=None, title="pyBiscuit QA") -> ViewerInputs:
    """Classify ``*.surf.gii`` / ``*.func.gii`` / ``*.label.gii`` files by name."""
    vi = ViewerInputs(volume=Path(volume) if volume else None, title=title)
    for p in map(Path, paths):
        hemi, mid = _hemi_kind(p.name)
        if p.name.endswith(".surf.gii"):
            vi.surfaces.append((hemi, mid, p))
        elif p.name.endswith(".func.gii"):
            vi.metrics.append((hemi, mid, p))
        elif p.name.endswith(".label.gii"):
            vi.labels.append((hemi, mid, p))
        else:
            raise ValueError(f"unsupported file {p.name!r} (want .surf.gii/.func.gii/.label.gii)")
    if not vi.surfaces:
        raise ValueError("need at least one *.surf.gii")
    return vi


def _payload(path: Path, embed: bool, base: Path) -> dict:
    if embed:
        return {"b64": base64.b64encode(path.read_bytes()).decode("ascii")}
    try:
        rel = path.resolve().relative_to(base.resolve())
    except ValueError:
        rel = Path(_relpath(path.resolve(), base.resolve()))
    return {"url": rel.as_posix()}


def _relpath(path: Path, base: Path) -> str:
    import os

    return os.path.relpath(path, base)


def _style_for(measure: str) -> dict:
    try:
        st = measure_style(measure)
    except KeyError:
        return {"cmap": "viridis", "min": None, "max": None, "label": measure}
    return {"cmap": st.cmap, "min": st.vmin, "max": st.vmax, "label": st.label}


def build_manifest(vi: ViewerInputs, embed: bool, base: Path) -> dict:
    metrics = []
    for hemi, name, p in vi.metrics:
        img = nib.load(str(p))
        ncol = len(img.darrays)
        metrics.append(
            {
                "hemi": hemi,
                "name": name,
                "file": p.name,
                "ncol": ncol,
                **_style_for(name),
                **_payload(p, embed, base),
            }
        )
    custom = {m["cmap"] for m in metrics} - _NV_BUILTIN
    return {
        "title": vi.title,
        "colormaps": {c: _sampled_colormap(c) for c in sorted(custom)},
        "niivue": niivue_info()["version"],
        "surfaces": [
            {"hemi": h, "kind": k, "file": p.name, **_payload(p, embed, base)}
            for h, k, p in vi.surfaces
        ],
        "metrics": metrics,
        "labels": [
            {"hemi": h, "name": n, "file": p.name, **_payload(p, embed, base)}
            for h, n, p in vi.labels
        ],
        "volume": (
            {"file": vi.volume.name, **_payload(vi.volume, embed, base)}
            if vi.volume is not None
            else None
        ),
    }


def write_viewer(out_html, vi: ViewerInputs, embed: bool = True) -> Path:
    """Write the viewer page to ``out_html`` and return its path."""
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(vi, embed, out_html.parent)
    page = (
        _TEMPLATE.replace("@@TITLE@@", html.escape(vi.title))
        .replace("@@VERSION@@", manifest["niivue"])
        .replace("@@MANIFEST@@", json.dumps(manifest).replace("</", "<\\/"))
        .replace("@@NIIVUE@@", _niivue_js())
    )
    out_html.write_text(page, encoding="utf-8")
    return out_html


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>@@TITLE@@</title>
<style>
  :root { --bg:#fff; --fg:#1b1f24; --muted:#59636e; --panel:#f3f4f6; --line:#d0d7de; }
  @media (prefers-color-scheme: dark) { :root { --bg:#14171a; --fg:#e6e8ea; --muted:#9aa4af;
    --panel:#1d2227; --line:#30363d; } }
  html, body { height:100%; margin:0; background:var(--bg); color:var(--fg);
    font:14px/1.4 system-ui, sans-serif; }
  #app { display:flex; flex-direction:column; height:100%; }
  header { display:flex; flex-wrap:wrap; gap:12px 18px; align-items:center; padding:8px 14px;
    background:var(--panel); border-bottom:1px solid var(--line); }
  header h1 { font-size:15px; margin:0 8px 0 0; }
  label { display:flex; align-items:center; gap:6px; color:var(--muted); }
  select, input[type=range] { font:inherit; }
  main { flex:1; min-height:0; position:relative; }
  canvas { width:100%; height:100%; display:block; }
  #status { position:absolute; left:10px; bottom:8px; color:var(--muted); font-size:12px; }
  #err { color:#b3261e; padding:6px 14px; display:none; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>@@TITLE@@</h1>
    <label>Surface <select id="surf"></select></label>
    <label>Measure <select id="metric"></select></label>
    <label>Labels <select id="label"></select></label>
    <label>Overlay opacity
      <input id="opacity" type="range" min="0" max="1" step="0.05" value="1"></label>
    <label id="hemis"></label>
    <label id="voltoggle" style="display:none">
      <input id="vol" type="checkbox" checked> Volume</label>
  </header>
  <div id="err"></div>
  <main><canvas id="gl"></canvas><div id="status"></div></main>
</div>
<script>@@NIIVUE@@</script>
<script>
const M = @@MANIFEST@@;
const $ = id => document.getElementById(id);
const status = t => { $("status").textContent = t; };
function bytes(item) {
  const bin = atob(item.b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}
async function toUrl(item) {
  if (item.b64) return URL.createObjectURL(new Blob([bytes(item)]));
  return item.url;
}
const nv = new niivue.Niivue({ backColor:[1,1,1,1], show3Dcrosshair:false, isOrientCube:true });
for (const [name, cm] of Object.entries(M.colormaps)) nv.addColormap(name, cm);
const state = { hemis: new Set(), layers: [] };

function fill(sel, entries, none) {
  sel.innerHTML = "";
  if (none) sel.add(new Option("none", ""));
  entries.forEach(([v, t]) => sel.add(new Option(t, v)));
}

let chain = Promise.resolve();
const rebuild = () => (chain = chain.then(doRebuild));
async function doRebuild() {
  const kind = $("surf").value, metric = $("metric").value, label = $("label").value;
  const op = parseFloat($("opacity").value);
  const meshes = [];
  for (const s of M.surfaces.filter(s => s.kind === kind && state.hemis.has(s.hemi))) {
    const layers = [];
    const m = M.metrics.find(x => x.hemi === s.hemi && x.name === metric);
    if (m) layers.push({ url: await toUrl(m), name: m.file, colormap: m.cmap, opacity: op,
      cal_min: m.min ?? undefined, cal_max: m.max ?? undefined,
      isTransparentBelowCalMin: false });
    const l = M.labels.find(x => x.hemi === s.hemi && x.name === label);
    if (l) layers.push({ url: await toUrl(l), name: l.file, opacity: op });
    meshes.push({ url: await toUrl(s), name: s.file, rgba255: [200, 200, 200, 255], layers });
  }
  while (nv.meshes.length) nv.removeMesh(nv.meshes[0]);
  if (meshes.length) await nv.loadMeshes(meshes);
  nv.updateGLVolume();
  status(M.niivue ? "NiiVue " + M.niivue + " (offline)" : "");
}

async function init() {
  try {
    await nv.attachTo("gl");
    const kinds = [...new Set(M.surfaces.map(s => s.kind))];
    const order = ["white", "pial", "mid", "inflated", "sphere", "sphere.reg"];
    kinds.sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
    fill($("surf"), kinds.map(k => [k, k]));
    // one entry per name (not per hemisphere); each hemisphere loads its own file
    const names = [...new Set(M.metrics.map(m => m.name))];
    fill($("metric"), names.map(n => [n, n]), true);
    const lnames = [...new Set(M.labels.map(l => l.name))];
    fill($("label"), lnames.map(n => [n, n]), true);
    const hs = [...new Set(M.surfaces.map(s => s.hemi))].sort();
    hs.forEach(h => state.hemis.add(h));
    $("hemis").innerHTML = hs.map(h =>
      `<label><input type="checkbox" data-h="${h}" checked> ${h}</label>`).join("");
    $("hemis").querySelectorAll("input").forEach(c => c.onchange = () => {
      c.checked ? state.hemis.add(c.dataset.h) : state.hemis.delete(c.dataset.h); rebuild(); });
    ["surf", "metric", "label", "opacity"].forEach(id => $(id).onchange = rebuild);
    if (M.volume) {
      $("voltoggle").style.display = "";
      await nv.loadVolumes([{ url: await toUrl(M.volume), name: M.volume.file }]);
      $("vol").onchange = () => { nv.setOpacity(0, $("vol").checked ? 1 : 0); };
      nv.setSliceType(nv.sliceTypeMultiplanar);
    } else {
      nv.setSliceType(nv.sliceTypeRender);
    }
    await rebuild();
  } catch (e) {
    $("err").style.display = "block"; $("err").textContent = "Viewer error: " + e;
    console.error(e);
  }
}
init();
</script>
</body>
</html>
"""
