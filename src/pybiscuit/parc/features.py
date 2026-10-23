"""pybiscuit.parc.features: feature extraction and label parsing for parcellation (B7.1).

Mirrors ``extract_features.m`` (spec §6.2) and label parsing from ``parc.m`` (spec §6.1).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from pybiscuit.mesh import Mesh
from pybiscuit.mesh.property import parc_normalize_feature, parc_white_mean_curvature

__all__ = [
    "extract_parc_features",
    "load_protocol_labels",
    "normalize_parc_feature",
]

normalize_parc_feature = parc_normalize_feature


def load_protocol_labels(label_csv_path: str | Path, hemi: str) -> tuple[np.ndarray, list[str]]:
    """Parse protocol ``label.csv`` (spec §6.1, ``parc.m:2-31``).

    Parameters
    ----------
    label_csv_path : str or Path
        Path to the protocol ``label.csv`` file.
    hemi : str
        ``'lh'`` / ``'Left'`` or ``'rh'`` / ``'Right'``.

    Returns
    -------
    label_ids : np.ndarray
        1D int64 array of protocol ROI IDs starting with ``0`` (unknown) at index 0,
        followed by hemisphere IDs in file order. Shape ``(nclass,)``.
    label_names : list[str]
        List of class label names starting with ``'Unknown'`` at index 0.
    """
    path = Path(label_csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Protocol label file not found: {path}")

    hemi_prefix = "left" if hemi.lower().startswith("l") else "right"

    ids: list[int] = [0]
    names: list[str] = ["Unknown"]

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        # Skip header
        next(reader, None)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            roi_id = int(row[1].strip())
            # strncmpi(d2c, 'Left', 4) or strncmpi(d2c, 'Right', 5)
            if name.lower().startswith(hemi_prefix):
                names.append(name)
                ids.append(roi_id)

    return np.array(ids, dtype=np.int64), names


def extract_parc_features(
    white_mesh: Mesh,
    ih_curv: np.ndarray,
    sulc_curv: np.ndarray,
    *,
    round_intermediate: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and normalise parcellation features (spec §6.2, ``extract_features.m``).

    Parameters
    ----------
    white_mesh : Mesh
        White matter surface.
    ih_curv : np.ndarray
        Inflated mean curvature feature (from FreeSurfer ``.inflated.H``).
    sulc_curv : np.ndarray
        Sulcal depth feature (from FreeSurfer ``.sulc``).
    round_intermediate : bool
        Whether to round intermediate ``MeshProperty`` output to 6 decimal places
        before negation and normalisation, matching legacy text-file round trips.

    Returns
    -------
    h_norm : np.ndarray
        Normalised ``-H`` curvature feature on the white surface.
    ih_norm : np.ndarray
        Normalised ``iH`` feature.
    sulc_norm : np.ndarray
        Normalised ``sulc`` feature.
    """
    h_norm = parc_white_mean_curvature(
        white_mesh,
        normalize=True,
        round_intermediate=round_intermediate,
    )
    ih_norm = normalize_parc_feature(ih_curv)
    sulc_norm = normalize_parc_feature(sulc_curv)
    return h_norm, ih_norm, sulc_norm
