"""End-to-end tests for parcellation pipeline against open-examples (B7.1, B7.3)."""

from pathlib import Path

import numpy as np
import pytest

from pybiscuit.io import read_vertex_txt
from pybiscuit.parc.pipeline import parcellate

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
_REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = (
    _REPO_ROOT.parent / "open-examples" / "0050952"
    if (_REPO_ROOT.parent / "open-examples" / "0050952").is_dir()
    else _REPO_ROOT / "open-examples" / "0050952"
)


def _dice_coefficient(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]
) -> dict[int, float]:
    dices = {}
    for lbl in labels:
        if lbl == 0:
            continue
        mask_true = y_true == lbl
        mask_pred = y_pred == lbl
        total = np.sum(mask_true) + np.sum(mask_pred)
        if total == 0:
            dices[lbl] = 1.0
        else:
            intersection = np.sum(mask_true & mask_pred)
            dices[lbl] = float(2.0 * intersection / total)
    return dices


@pytest.mark.reference
@pytest.mark.skipif(not EXAMPLES_DIR.is_dir(), reason="open-examples/0050952 not found")
def test_parcellate_braincolor_lh(tmp_path):
    golden_path = EXAMPLES_DIR / "roi" / "lh.parc.braincolor.txt"
    assert golden_path.is_file(), f"Golden file not found: {golden_path}"
    golden_labels = read_vertex_txt(golden_path, dtype=np.int64)

    pred_labels = parcellate(
        subject_dir=EXAMPLES_DIR,
        hemi="lh",
        protocol="braincolor",
        out_dir=tmp_path,
        degree=0,
    )

    written_labels = read_vertex_txt(tmp_path / "lh.parc.braincolor.txt", dtype=np.int64)
    assert np.array_equal(pred_labels, written_labels)
    assert len(pred_labels) == len(golden_labels)

    agreement = float(np.mean(pred_labels == golden_labels))
    unique_labels = sorted(set(golden_labels.tolist()))
    dices = _dice_coefficient(golden_labels, pred_labels, unique_labels)
    min_dice = min(dices.values())
    median_dice = float(np.median(list(dices.values())))

    print(
        f"\nBrainColor lh agreement: {agreement * 100:.3f}% "
        f"(min Dice: {min_dice:.4f}, median Dice: {median_dice:.4f})"
    )

    # Acceptance criterion: >= 99.5% agreement, median Dice >= 0.99
    assert agreement >= 0.995, f"Agreement {agreement * 100:.3f}% below 99.5%"
    assert median_dice >= 0.99, f"Median Dice {median_dice:.4f} below 0.99"
    assert min_dice >= 0.95, f"Min Dice {min_dice:.4f} below 0.95"


@pytest.mark.reference
@pytest.mark.skipif(not EXAMPLES_DIR.is_dir(), reason="open-examples/0050952 not found")
def test_parcellate_dkt31_lh(tmp_path):
    golden_path = EXAMPLES_DIR / "roi" / "lh.parc.dkt31.txt"
    assert golden_path.is_file(), f"Golden file not found: {golden_path}"
    golden_labels = read_vertex_txt(golden_path, dtype=np.int64)

    pred_labels = parcellate(
        subject_dir=EXAMPLES_DIR,
        hemi="lh",
        protocol="dkt31",
        out_dir=tmp_path,
        degree=0,
    )

    written_labels = read_vertex_txt(tmp_path / "lh.parc.dkt31.txt", dtype=np.int64)
    assert np.array_equal(pred_labels, written_labels)
    assert len(pred_labels) == len(golden_labels)

    agreement = float(np.mean(pred_labels == golden_labels))
    unique_labels = sorted(set(golden_labels.tolist()))
    dices = _dice_coefficient(golden_labels, pred_labels, unique_labels)
    min_dice = min(dices.values())
    median_dice = float(np.median(list(dices.values())))

    print(
        f"\nDKT31 lh agreement: {agreement * 100:.3f}% "
        f"(min Dice: {min_dice:.4f}, median Dice: {median_dice:.4f})"
    )

    assert agreement >= 0.995, f"Agreement {agreement * 100:.3f}% below 99.5%"
    assert median_dice >= 0.99, f"Median Dice {median_dice:.4f} below 0.99"
    assert min_dice >= 0.95, f"Min Dice {min_dice:.4f} below 0.95"
