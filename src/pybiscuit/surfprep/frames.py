"""Alignment frames for the cutting plane (open question Q4; spec §4.2 step 2).

BISCUIT places the cutting plane with an MNI copy of the inner surface that has
the same vertex order as the native one.  Only its *orientation* matters: the
candidate planes are ``x = mean(x) + r`` with ``|r| <= 0.5`` mm, so any rigid
frame whose x axis is perpendicular to the mid-sagittal fissure serves.  Three
ways to get such a frame, each returning reference vertices for
:func:`pybiscuit.surfprep.hemisphere.separate_hemispheres`:

* :func:`reference_from_mni_vertices`  the MNI copy itself (identity);
* :func:`reference_from_affine`        legacy ``Surfaces_MNI/aff.txt``;
* :func:`reference_from_labels`        PyMaCRUISE default (Q4): a plane fitted to
  the ``SEG_refine`` left/right labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# SEG_refine / BrainCOLOR ids by side (spec §4.5 ``filled`` table: 255 = left, 127 = right).
LEFT_LABELS = (37, 45, 52, 56, 58, 60, 62)
RIGHT_LABELS = (36, 44, 51, 55, 57, 59, 61)
CORTEX_MIN = 100  # labels >= 100: odd = left cortex, even = right cortex


def read_aff(path) -> np.ndarray:
    """Read a 4x4 ``aff.txt``.  In the legacy sample it maps MNI -> native."""
    a = np.loadtxt(path, dtype=np.float64).reshape(4, 4)
    return a


def reference_from_mni_vertices(vertices) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float64)


def reference_from_affine(vertices, aff, *, aff_maps="mni_to_native") -> np.ndarray:
    """Native vertices -> the MNI-like frame.

    ``aff`` is the 4x4 of ``aff.txt`` (``aff_maps="mni_to_native"``, as verified
    on the legacy sample to 5e-6 mm) or its inverse (``"native_to_mni"``).
    """
    a = np.asarray(aff, dtype=np.float64)
    if aff_maps == "mni_to_native":
        a = np.linalg.inv(a)
    elif aff_maps != "native_to_mni":
        raise ValueError("aff_maps must be 'mni_to_native' or 'native_to_mni'")
    return np.asarray(vertices, dtype=np.float64) @ a[:3, :3].T + a[:3, 3]


def rotation_to_x(normal) -> np.ndarray:
    """Smallest rotation matrix taking ``normal`` to +x (Rodrigues)."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    x = np.array([1.0, 0.0, 0.0])
    c = float(n @ x)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([-1.0, -1.0, 1.0])
    k = np.cross(n, x)
    s2 = float(k @ k)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + kx + kx @ kx * ((1 - c) / s2)


def side_masks(data) -> tuple[np.ndarray, np.ndarray]:
    """Boolean left/right masks from SEG_refine label values."""
    d = np.asarray(data)
    d = np.rint(d).astype(np.int64) if d.dtype.kind == "f" else d.astype(np.int64)
    left = np.isin(d, LEFT_LABELS) | ((d >= CORTEX_MIN) & (d % 2 == 1))
    right = np.isin(d, RIGHT_LABELS) | ((d >= CORTEX_MIN) & (d % 2 == 0))
    return left, right


def fit_midline_normal(left_points, right_points) -> np.ndarray:
    """Unit normal of the plane separating two point clouds (Fisher/LDA direction).

    Pooled within-class covariance, regularised by 1e-6 of its trace; points
    from the left cloud toward the right cloud (so +x = right in a RAS frame).
    """
    lp = np.asarray(left_points, dtype=np.float64)
    rp = np.asarray(right_points, dtype=np.float64)
    if lp.shape[0] < 10 or rp.shape[0] < 10:
        raise ValueError("need left and right label voxels to fit the midline")
    ml, mr = lp.mean(axis=0), rp.mean(axis=0)
    sw = np.cov((lp - ml).T) * (lp.shape[0] - 1) + np.cov((rp - mr).T) * (rp.shape[0] - 1)
    sw /= lp.shape[0] + rp.shape[0] - 2
    sw += np.eye(3) * (1e-6 * np.trace(sw))
    n = np.linalg.solve(sw, mr - ml)
    n /= np.linalg.norm(n)
    if n @ (mr - ml) < 0:
        n = -n
    return n


def midline_normal_from_seg(seg_path_or_img) -> np.ndarray:
    """Midline normal (scanner RAS) from a ``SEG_refine`` NIfTI path or nibabel image."""
    import nibabel as nib

    img = (
        nib.load(str(seg_path_or_img))
        if isinstance(seg_path_or_img, str | Path)
        else (seg_path_or_img)
    )
    data = np.asarray(img.dataobj)
    left, right = side_masks(data)
    aff = img.affine

    def pts(mask):
        ijk = np.argwhere(mask)
        return ijk @ aff[:3, :3].T + aff[:3, 3]

    return fit_midline_normal(pts(left), pts(right))


def reference_from_labels(vertices, seg_path_or_img) -> np.ndarray:
    """Q4 default: rotate native vertices so the label-fitted midline normal is +x."""
    v = np.asarray(vertices, dtype=np.float64)
    rot = rotation_to_x(midline_normal_from_seg(seg_path_or_img))
    return (v - v.mean(axis=0)) @ rot.T
