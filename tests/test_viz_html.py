"""Offline NiiVue viewer generation (no browser; page behaviour is checked manually, see report)."""

import hashlib
import json
import re

import nibabel as nib
import numpy as np
import pytest

from fixtures import icosphere
from pybiscuit.io import read_protocol_table, write_func_gii, write_label_gii, write_surf_gii
from pybiscuit.viz import html as H


@pytest.fixture
def files(tmp_path):
    m = icosphere(2)
    t = tmp_path / "parc"
    t.mkdir()
    (t / "label.csv").write_text("label,ID\nfoo,3\nbar,7\n")
    (t / "lut.csv").write_text("ID,R,G,B\n3,10,20,30\n7,200,100,50\n")
    out = tmp_path / "run"
    out.mkdir()
    paths = []
    for h in ("lh", "rh"):
        for k in ("white", "inflated"):
            write_surf_gii(out / f"{h}.{k}.surf.gii", m, h, k)
            paths.append(out / f"{h}.{k}.surf.gii")
        write_func_gii(out / f"{h}.thickness.func.gii", m.vertices[:, 2] + 2, h, ["thickness"])
        write_func_gii(out / f"{h}.curv.func.gii", m.vertices[:, 0] * 0.1, h, ["curv"])
        write_label_gii(
            out / f"{h}.dkt31.label.gii",
            np.where(m.vertices[:, 0] > 0, 3, 7),
            read_protocol_table(t),
            h,
        )
        paths += [
            out / f"{h}.thickness.func.gii",
            out / f"{h}.curv.func.gii",
            out / f"{h}.dkt31.label.gii",
        ]
    return out, paths


def _manifest(text):
    m = re.search(r"const M = (\{.*?\});\nconst \$", text, re.S)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_bundled_niivue_matches_recorded_hash():
    info = H.niivue_info()
    assert info["version"] == "0.69.0" and info["license"] == "BSD-2-Clause"
    sha = hashlib.sha256((H.DATA_DIR / info["bundled_file"]).read_bytes()).hexdigest()
    assert sha == info["bundled_sha256"]


def test_embedded_viewer_is_self_contained(files):
    out, paths = files
    vi = H.collect_inputs(paths, title="t <x>")
    page = H.write_viewer(out / "qa" / "index.html", vi).read_text()
    # no external resources: scripts are inline, no href/src to http(s) in our own markup
    own = page.split("<script>")[0]
    assert not re.search(r'(?:src|href)\s*=\s*"?https?:', own)
    assert "<script src" not in page and "<link " not in page
    M = _manifest(page)
    assert M["niivue"] == "0.69.0" and all("b64" in s for s in M["surfaces"])
    assert {s["kind"] for s in M["surfaces"]} == {"white", "inflated"}
    assert {m["name"] for m in M["metrics"]} == {"thickness", "curv"}
    assert {m["cmap"] for m in M["metrics"]} == {"viridis", "RdBu_r"}
    assert set(M["colormaps"]) == {"RdBu_r"} and len(M["colormaps"]["RdBu_r"]["R"]) == 32
    thick = next(m for m in M["metrics"] if m["name"] == "thickness")
    assert (thick["min"], thick["max"]) == (1.0, 5.0)  # fixed limits from the PDF
    import base64

    surf = next(s for s in M["surfaces"] if s["hemi"] == "rh" and s["kind"] == "white")
    (out / "dec.surf.gii").write_bytes(base64.b64decode(surf["b64"]))
    assert nib.load(out / "dec.surf.gii").darrays[0].data.shape[1] == 3
    assert "&lt;x&gt;" in page  # title is escaped


def test_linked_viewer_uses_relative_urls(files):
    out, paths = files
    vi = H.collect_inputs(paths)
    page = H.write_viewer(out / "qa" / "index.html", vi, embed=False).read_text()
    M = _manifest(page)
    assert all("b64" not in s and s["url"].startswith("../") for s in M["surfaces"])
    assert len(page) < 4_000_000  # just NiiVue + manifest, no data


def test_volume_and_input_validation(files, tmp_path):
    out, paths = files
    vol = tmp_path / "t1.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), np.eye(4)), vol)
    vi = H.collect_inputs(paths, volume=vol)
    M = _manifest(H.write_viewer(out / "v.html", vi).read_text())
    assert M["volume"]["file"] == "t1.nii.gz" and "b64" in M["volume"]
    with pytest.raises(ValueError):
        H.collect_inputs([out / "lh.curv.func.gii"])  # no surface
    with pytest.raises(ValueError):
        H.collect_inputs([tmp_path / "notes.txt"])
