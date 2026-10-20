import time

import numpy as np
import pytest

from conftest import OPT
from pybiscuit.io.txt import read_vertex_txt, write_vertex_txt


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_roundtrip(tmp_path, dtype):
    rng = np.random.default_rng(0)
    v = (rng.normal(size=1000) * 100).astype(dtype)
    p = tmp_path / "v.txt"
    write_vertex_txt(v, p)
    out = read_vertex_txt(p, dtype=dtype, n_vertices=1000)
    np.testing.assert_array_equal(out, v)


def test_count_mismatch_and_garbage(tmp_path):
    p = tmp_path / "v.txt"
    p.write_text("1\n2\n3\n")
    with pytest.raises(ValueError):
        read_vertex_txt(p, n_vertices=4)
    p.write_text("1\nabc\n3\n")
    with pytest.raises(ValueError):
        read_vertex_txt(p)
    p.write_text("")
    assert read_vertex_txt(p).size == 0


def test_rejects_2d(tmp_path):
    with pytest.raises(ValueError):
        write_vertex_txt(np.zeros((3, 2)), tmp_path / "x.txt")


@pytest.mark.reference
@pytest.mark.parametrize("hemi", ["lh", "rh"])
@pytest.mark.parametrize("stat", ["mean", "var"])
def test_reference_reg_txt(hemi, stat):
    files = sorted((OPT / "reg").glob(f"*/{hemi}.*.{stat}.txt"))
    assert files
    for f in files:
        t = time.perf_counter()
        v = read_vertex_txt(f)
        assert v.shape == (163842,), f
        assert np.isfinite(v).all(), f
        assert time.perf_counter() - t < 5
