"""pybiscuit.sphere: icospheres and MeshConv operators."""

from . import remesh, spharm  # noqa: F401
from .aabb import SphereAABB, barycentric, cart2sph, closest_face
from .icosphere import (
    MAX_LEVEL,
    SCHEMES,
    SHIPPED_SPHERES,
    icosphere,
    icosphere_counts,
    quantize_shipped,
)
from .operators import (
    FIELDS,
    SafeUnpicklingError,
    build_operators,
    compare_operators,
    load_operators_npz,
    safe_load_pickle,
    save_operators_npz,
)

__all__ = [
    "SphereAABB",
    "barycentric",
    "cart2sph",
    "closest_face",
    "remesh",
    "spharm",
    "FIELDS",
    "MAX_LEVEL",
    "SCHEMES",
    "SHIPPED_SPHERES",
    "SafeUnpicklingError",
    "build_operators",
    "compare_operators",
    "icosphere",
    "icosphere_counts",
    "load_operators_npz",
    "quantize_shipped",
    "safe_load_pickle",
    "save_operators_npz",
]
