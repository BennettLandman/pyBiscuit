import numpy as np

from pybiscuit.io.gifti import read_gifti, write_gifti
from pybiscuit.mesh.model import Mesh


def test_gifti_roundtrip(tmp_path):
    v = np.array([[0, 0, 0], [1.5, 0, 0], [0, 2.25, 0], [0, 0, -3.125]], float)  # float32-exact
    f = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.int32)
    m = Mesh(
        v,
        f,
        {"thickness": np.array([1.0, 2.5, 3.0, 0.25]), "label": np.array([1, 2, 3, 4])},
    )
    write_gifti(tmp_path / "m.surf.gii", m)
    r = read_gifti(tmp_path / "m.surf.gii")
    np.testing.assert_array_equal(r.vertices, v)
    np.testing.assert_array_equal(r.faces, f)
    np.testing.assert_array_equal(r.point_data["thickness"], m.point_data["thickness"])
    np.testing.assert_array_equal(r.point_data["label"], m.point_data["label"])
