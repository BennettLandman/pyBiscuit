import numpy as np
import pytest

from fixtures import icosphere

pytest.importorskip("pyvista")
pytest.importorskip("matplotlib")

from pybiscuit.cli import main  # noqa: E402
from pybiscuit.io import (  # noqa: E402
    read_protocol_table,
    write_func_gii,
    write_label_gii,
    write_surf_gii,
    write_vertex_txt,
)
from pybiscuit.viz import cli as C  # noqa: E402
from pybiscuit.viz import render as R  # noqa: E402


@pytest.fixture
def run(tmp_path):
    m = icosphere(2)
    parc = tmp_path / "parc"
    parc.mkdir()
    (parc / "label.csv").write_text("label,ID\nfoo,3\nbar,7\n")
    (parc / "lut.csv").write_text("ID,R,G,B\n3,255,0,0\n7,0,0,255\n")
    d = tmp_path / "run"
    d.mkdir()
    write_surf_gii(d / "lh.white.surf.gii", m, "lh", "white")
    write_surf_gii(d / "lh.inflated.surf.gii", m, "lh", "inflated")
    write_func_gii(d / "lh.thickness.func.gii", m.vertices[:, 2] + 3, "lh", ["thickness"])
    write_label_gii(
        d / "lh.dkt31.label.gii",
        np.where(m.vertices[:, 0] > 0, 3, 7),
        read_protocol_table(parc),
        "lh",
    )
    write_vertex_txt(np.where(m.vertices[:, 1] > 0, 3, 7), d / "lh.aparc.txt")
    return d, parc


def test_load_scenes(run):
    d, parc = run
    files = [
        d / n
        for n in (
            "lh.inflated.surf.gii",
            "lh.white.surf.gii",
            "lh.thickness.func.gii",
            "lh.dkt31.label.gii",
            "lh.aparc.txt",
        )
    ]
    sc = C.load_scenes(files, parc=str(parc))["lh"]
    assert sc.surface_names() == ["white", "inflated"]
    assert list(sc.metrics) == ["thickness"]
    assert set(sc.labels) == {"dkt31", "aparc"}
    assert sc.labels["aparc"][1].rgb(3) == (255, 0, 0)
    assert [o[1] for o in sc.overlays()] == ["none", "thickness", "dkt31", "aparc"]
    # without --parc a .txt is a measure
    sc2 = C.load_scenes(files)["lh"]
    assert "aparc" in sc2.metrics and set(sc2.labels) == {"dkt31"}


def test_load_scenes_errors(run, tmp_path):
    d, _ = run
    with pytest.raises(SystemExit):
        C.load_scenes([d / "lh.thickness.func.gii"])  # no surface
    with pytest.raises(SystemExit):
        C.load_scenes([tmp_path / "missing.surf.gii"])
    (tmp_path / "x.surf.gii").write_bytes(b"")
    with pytest.raises(SystemExit):
        C.load_scenes([d / "lh.white.surf.gii", tmp_path / "x.surf.gii"])  # no hemisphere
    write_vertex_txt(np.zeros(5), d / "lh.short.txt")
    with pytest.raises(SystemExit):
        C.load_scenes([d / "lh.white.surf.gii", d / "lh.short.txt"])  # wrong length


def test_view_state_cycles(run):
    d, _ = run
    sc = C.load_scenes(
        [
            d / "lh.white.surf.gii",
            d / "lh.inflated.surf.gii",
            d / "lh.thickness.func.gii",
            d / "lh.dkt31.label.gii",
        ]
    )["lh"]
    st = C.ViewState(sc)
    assert st.current() == ("white", ("none", "none"))
    assert st.cycle_surface() == "inflated" and st.cycle_surface() == "white"
    assert st.cycle_overlay() == ("metric", "thickness")
    rgb, style = st.colors()
    assert style.limits == (1.0, 5.0) and rgb.shape == (sc.surfaces["white"].n_vertices, 3)
    assert st.cycle_overlay() == ("label", "dkt31")
    rgb, style = st.colors()
    assert style is None and {tuple(np.round(c * 255)) for c in rgb} == {(255, 0, 0), (0, 0, 255)}
    assert st.cycle_overlay() == ("none", "none")
    assert st.cycle_overlay(-1) == ("label", "dkt31")


def test_render_cli_requires_known_views(run, capsys):
    d, _ = run
    with pytest.raises(SystemExit):
        main(["render", str(d / "lh.white.surf.gii"), "--out", str(d / "o"), "--views", "bogus"])


@pytest.mark.skipif(not R.gl_available(), reason="no off-screen OpenGL context")
def test_render_cli_writes_panels(run, capsys):
    d, _ = run
    out = d / "png"
    files = [
        str(d / n)
        for n in (
            "lh.white.surf.gii",
            "lh.inflated.surf.gii",
            "lh.thickness.func.gii",
            "lh.dkt31.label.gii",
        )
    ]
    rc = main(
        ["render", *files, "--out", str(out), "--tile", "64", "--html", str(out / "index.html")]
    )
    assert rc == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["index.html", "lh.dkt31.png", "lh.surfaces.png", "lh.thickness.png"]
    assert (out / "lh.thickness.png").read_bytes()[:4] == b"\x89PNG"


@pytest.mark.skipif(not R.gl_available(), reason="no off-screen OpenGL context")
def test_view_screenshot(run):
    d, _ = run
    shot = d / "v.png"
    rc = main(
        [
            "view",
            str(d / "lh.white.surf.gii"),
            str(d / "lh.thickness.func.gii"),
            "--screenshot",
            str(shot),
        ]
    )
    assert rc == 0 and shot.stat().st_size > 1000
