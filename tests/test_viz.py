"""Viz tests. Pure-numpy parts always run; rendering tests skip without an off-screen GL context."""

from pathlib import Path

import numpy as np
import pytest

from fixtures import folded_sphere, icosphere

pytest.importorskip("pyvista")
pytest.importorskip("matplotlib")

from pybiscuit.io.labels import LabelTable  # noqa: E402
from pybiscuit.viz import MEASURES, VIEWS, camera_for, measure_style  # noqa: E402
from pybiscuit.viz import render as R  # noqa: E402

GOLD = Path(__file__).parent / "data" / "viz"
needs_gl = pytest.mark.skipif(not R.gl_available(), reason="no off-screen OpenGL context")


# --- cameras / colour conventions (no GL) -------------------------------------------------


def test_camera_hemisphere_aware():
    b = (-1, 1, -2, 2, -3, 3)
    for hemi, sx in (("lh", -1), ("rh", 1)):
        pos, focal, up, _ = camera_for("lateral", hemi, b)
        assert np.sign((pos - focal)[0]) == sx and np.allclose(up, (0, 0, 1))
        pos, focal, _, _ = camera_for("medial", hemi, b)
        assert np.sign((pos - focal)[0]) == -sx
    for view, axis, sign in (
        ("dorsal", 2, 1),
        ("ventral", 2, -1),
        ("anterior", 1, 1),
        ("posterior", 1, -1),
    ):
        pos, focal, up, _ = camera_for(view, "lh", b)
        d = pos - focal
        assert np.argmax(np.abs(d)) == axis and np.sign(d[axis]) == sign
        assert abs(up @ (d / np.linalg.norm(d))) < 1e-12
    with pytest.raises(ValueError):
        camera_for("sideways", "lh", b)
    with pytest.raises(ValueError):
        camera_for("lateral", "xx", b)
    assert set(VIEWS) == {"lateral", "medial", "dorsal", "ventral", "anterior", "posterior"}


def test_camera_scale_fits_extent():
    _, _, _, s = camera_for("lateral", "lh", (-1, 1, -10, 10, -3, 3), margin=1.0)
    assert s == 10  # y is the widest extent seen from the side
    _, _, _, s = camera_for("dorsal", "lh", (-1, 1, -10, 10, -30, 30), margin=1.0)
    assert s == 10


def test_fixed_limits_match_pdf():
    expect = {
        "mean_curvature": (-0.05, 0.05),
        "shape_index": (-1, 1),
        "thickness": (1, 5),
        "sulcal_depth": (0, 30),
        "lgi": (2, 10),
        "sci": (0, 1),
    }
    for name, lim in expect.items():
        assert MEASURES[name].limits == lim and MEASURES[name].source == "pdf"
    assert measure_style("lh.LGI") is MEASURES["lgi"]
    assert measure_style("SI") is MEASURES["shape_index"]
    assert measure_style("thickness").cmap == "viridis" and measure_style("curv").cmap == "RdBu_r"
    with pytest.raises(KeyError):
        measure_style("mystery")
    st = measure_style("mystery", np.arange(101.0))
    assert st.limits == (2.0, 98.0)


def test_color_helpers():
    st = MEASURES["sci"]
    rgb = R.scalar_colors([-5, 0.0, 1.0, 9, np.nan], st)
    np.testing.assert_allclose(rgb[0], rgb[1])  # clipped to the limit
    np.testing.assert_allclose(rgb[2], rgb[3])
    assert not np.allclose(rgb[1], rgb[2])
    np.testing.assert_allclose(rgb[4], R.NAN_COLOR)
    t = LabelTable({1: ("a", (255, 0, 0)), 2: ("b", (0, 0, 255))})
    c = R.label_colors([1, 2, 0, 7], t)
    np.testing.assert_allclose(c[0], (1, 0, 0))
    np.testing.assert_allclose(c[1], (0, 0, 1))
    np.testing.assert_allclose(c[2], R.UNDERLAY_GRAY)
    np.testing.assert_allclose(c[3], R.UNDERLAY_GRAY)


# --- rendering ----------------------------------------------------------------------------


def _scalar_mesh():
    m = folded_sphere(level=3)
    return m, m.vertices[:, 2]  # z as an analytic scalar


def _close(img, name, tol=0.03):
    """Perceptual-ish compare: 4x4 block means, mean abs difference below ``tol`` (0..1)."""
    from PIL import Image

    path = GOLD / name
    if not path.exists():  # first run: create the golden
        GOLD.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(path)
    gold = np.asarray(Image.open(path).convert("RGB"))
    assert gold.shape == img.shape

    def blocks(a):
        h, w, _ = a.shape
        return a[: h // 4 * 4, : w // 4 * 4].reshape(h // 4, 4, w // 4, 4, 3).mean((1, 3)) / 255

    return float(np.abs(blocks(gold) - blocks(img)).mean()) < tol


@needs_gl
def test_render_scalar_snapshots():
    m, z = _scalar_mesh()
    st = measure_style("curv")
    for view in ("lateral", "dorsal"):
        img = R.render_surface(m, "lh", view, scalars=z * 0.5, style=st, size=(96, 96))
        assert img.shape == (96, 96, 3) and img.dtype == np.uint8
        assert _close(img, f"scalar_{view}.png")


@needs_gl
def test_render_is_deterministic_and_view_dependent():
    m, z = _scalar_mesh()
    a = R.render_surface(m, "lh", "lateral", scalars=z, measure="curv", size=(64, 64))
    b = R.render_surface(m, "lh", "lateral", scalars=z, measure="curv", size=(64, 64))
    np.testing.assert_array_equal(a, b)
    c = R.render_surface(m, "lh", "dorsal", scalars=z, measure="curv", size=(64, 64))
    assert np.abs(a.astype(int) - c).mean() > 1


@needs_gl
def test_render_labels_and_curves():
    m = icosphere(3)
    t = LabelTable({1: ("a", (255, 0, 0)), 2: ("b", (0, 0, 255))})
    lab = np.where(m.vertices[:, 0] > 0, 1, 2)
    img = R.render_surface(m, "rh", "lateral", labels=lab, table=t, size=(96, 96))
    assert _close(img, "labels_lateral.png")
    # lateral rh camera sits at +x: label 1 (x>0) faces it, so red dominates over blue
    assert img[..., 0].astype(int).sum() - img[..., 2].astype(int).sum() > 0
    th = np.linspace(0, np.pi, 40)
    curve = np.stack([1.02 * np.sin(th), np.zeros_like(th), 1.02 * np.cos(th)], 1)
    with_curve = R.render_surface(
        m, "rh", "lateral", labels=lab, table=t, curves=[curve], curve_radius=0.03, size=(96, 96)
    )
    assert not np.array_equal(img, with_curve)
    assert _close(with_curve, "labels_curve_lateral.png")


@needs_gl
def test_render_panel_png(tmp_path):
    m, z = _scalar_mesh()
    out = tmp_path / "p.png"
    canvas = R.render_panel(
        [("a", m), ("b", icosphere(2))],
        "lh",
        ["lateral", "medial", "dorsal"],
        scalars={"a": z, "b": icosphere(2).vertices[:, 2]},
        measure="curv",
        tile_size=(64, 64),
        path=out,
    )
    from PIL import Image

    im = Image.open(out)
    assert im.size[0] >= 3 * 64 and im.size[1] >= 2 * 64
    assert canvas.shape[2] == 3


def test_gl_context_when_required():
    """The CI viz job sets PYBISCUIT_REQUIRE_GL=1 so a missing context fails instead of skipping."""
    import os

    if os.environ.get("PYBISCUIT_REQUIRE_GL"):
        assert R.gl_available(), "no off-screen OpenGL context in the viz CI job"
