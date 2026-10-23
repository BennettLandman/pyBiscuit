"""Unit tests for parcellation feature extraction and label loading (B7.1)."""

from pathlib import Path

import numpy as np
import pytest

from pybiscuit.parc.features import (
    load_protocol_labels,
    normalize_parc_feature,
)

DEV_ROOT = Path("/Users/landmaba/dev/biscuit")
ASSET_DIR = DEV_ROOT / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit" / "parc"


def test_normalize_parc_feature_synthetic():
    rng = np.random.default_rng(42)
    x = rng.normal(loc=10.0, scale=5.0, size=1000)
    norm = normalize_parc_feature(x)

    # Soft clamp at +-3 keeps all values strictly inside (-4, 4)
    assert np.all(norm > -4.0)
    assert np.all(norm < 4.0)

    # Monotonicity test
    sorted_idx = np.argsort(x)
    assert np.all(np.diff(norm[sorted_idx]) >= 0)


@pytest.mark.skipif(not ASSET_DIR.is_dir(), reason="Reference assets not present")
def test_load_protocol_labels_braincolor():
    csv_path = ASSET_DIR / "braincolor" / "label.csv"
    ids_lh, names_lh = load_protocol_labels(csv_path, "lh")
    ids_rh, names_rh = load_protocol_labels(csv_path, "rh")

    assert len(ids_lh) == 50
    assert len(ids_rh) == 50
    assert ids_lh[0] == 0
    assert names_lh[0] == "Unknown"
    assert ids_rh[0] == 0
    assert names_rh[0] == "Unknown"

    # Hemispheric label prefix checks
    assert all(name.lower().startswith("left") for name in names_lh[1:])
    assert all(name.lower().startswith("right") for name in names_rh[1:])


@pytest.mark.skipif(not ASSET_DIR.is_dir(), reason="Reference assets not present")
def test_load_protocol_labels_dkt31():
    csv_path = ASSET_DIR / "dkt31" / "label.csv"
    ids_lh, names_lh = load_protocol_labels(csv_path, "lh")
    ids_rh, names_rh = load_protocol_labels(csv_path, "rh")

    assert len(ids_lh) == 32
    assert len(ids_rh) == 32
    assert ids_lh[0] == 0
    assert names_lh[0] == "Unknown"
    assert ids_rh[0] == 0
    assert names_rh[0] == "Unknown"

    assert all(name.lower().startswith("left") for name in names_lh[1:])
    assert all(name.lower().startswith("right") for name in names_rh[1:])
