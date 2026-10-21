"""Camera presets, hemisphere-aware (RAS+ mm: +x right, +y anterior, +z superior).

Lateral looks at a hemisphere from outside (left hemisphere: camera at -x); medial from the
midline side. All views use a parallel projection so figures are comparable across subjects.
"""

from __future__ import annotations

import numpy as np

VIEWS = ("lateral", "medial", "dorsal", "ventral", "anterior", "posterior")

# direction from the focal point towards the camera, and the up vector (RAS+)
_FIXED = {
    "dorsal": ((0, 0, 1), (0, 1, 0)),
    "ventral": ((0, 0, -1), (0, 1, 0)),
    "anterior": ((0, 1, 0), (0, 0, 1)),
    "posterior": ((0, -1, 0), (0, 0, 1)),
}


def _hemi_sign(hemi: str) -> int:
    h = hemi.lower()[:1]
    if h not in ("l", "r"):
        raise ValueError(f"hemisphere must be lh or rh, got {hemi!r}")
    return -1 if h == "l" else 1


def view_direction(view: str, hemi: str) -> tuple[np.ndarray, np.ndarray]:
    """Return unit ``(toward_camera, up)`` vectors for a preset."""
    if view == "lateral":
        d, up = (_hemi_sign(hemi), 0, 0), (0, 0, 1)
    elif view == "medial":
        d, up = (-_hemi_sign(hemi), 0, 0), (0, 0, 1)
    elif view in _FIXED:
        d, up = _FIXED[view]
    else:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    return np.asarray(d, float), np.asarray(up, float)


def camera_for(view: str, hemi: str, bounds, margin: float = 1.05):
    """Camera for ``bounds`` (xmin, xmax, ymin, ymax, zmin, zmax).

    Returns ``(position, focal_point, up, parallel_scale)`` where ``parallel_scale`` is the
    half-height of the view in mm (VTK convention), chosen so the projected extent fits.
    """
    b = np.asarray(bounds, float).reshape(3, 2)
    center = b.mean(axis=1)
    half = (b[:, 1] - b[:, 0]) / 2
    d, up = view_direction(view, hemi)
    right = np.cross(up, d)  # image-plane horizontal axis
    # half-extent of the box projected on the up axis (vertical) and right axis
    vert = float(np.abs(up) @ half)
    horiz = float(np.abs(right) @ half)
    dist = 4.0 * float(np.linalg.norm(half)) + 1.0
    return center + d * dist, center, up, margin * max(vert, horiz)
