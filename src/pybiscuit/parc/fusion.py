"""pybiscuit.parc.fusion: per-model refinement and multi-model fusion (B7.3).

Mirrors ``parc.m:89-125`` (spec §6.6):
1. For each fold/model:
   - Run ``gcut`` and ``roi_correction`` on raw scores.
   - Compute per-vertex softmax probability of the chosen class.
2. If multiple models (e.g. DKT31 5 folds):
   - Softmax-weighted voting across models per class.
   - Renormalise votes and take ``log(prob + eps)``.
   - Second pass of ``gcut`` and ``roi_correction``.
3. Map 0-based class indices to protocol ROI IDs via ``label[pred]``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .cleanup import roi_correction
from .graphcut import gcut

__all__ = [
    "FusionRule",
    "PENDING",
    "fuse",
    "fuse_folds",
    "refine_fold",
]

FusionRule = Callable[[Sequence[np.ndarray]], np.ndarray]
PENDING = "DKT31 fusion rule pending B7.1 (parc.m is encrypted; see docs/open-questions.md Q7)"


def fuse(outputs: Sequence[np.ndarray], rule: FusionRule | None = None) -> np.ndarray:
    """Legacy helper: combine per-model outputs with an explicit ``rule``."""
    if rule is None:
        raise NotImplementedError(PENDING)
    arrs = [np.asarray(o) for o in outputs]
    if len({a.shape for a in arrs}) != 1:
        raise ValueError("all model outputs must have the same shape")
    return rule(arrs)


def refine_fold(
    vertices: np.ndarray,
    faces: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine raw class scores of a single model fold via graph cut and ROI cleanup.

    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices, shape ``(nV, 3)``.
    faces : np.ndarray
        Mesh triangular faces, shape ``(nF, 3)``.
    scores : np.ndarray
        Raw per-class scores on native vertices, shape ``(nclass, nV)``.

    Returns
    -------
    pred : np.ndarray
        0-based class indices for each vertex, shape ``(nV,)`` int64.
    pred_prob : np.ndarray
        Softmax probability of the predicted class for each vertex, shape ``(nV,)`` float64.
    """
    scores = np.asarray(scores, dtype=np.float32)
    nclass, nv = scores.shape

    # 1. Graph cut + ROI cleanup
    pred = gcut(vertices, faces, scores)
    pred = roi_correction(pred, vertices, faces, scores)

    # 2. Softmax probabilities
    max_scores = np.max(scores, axis=0, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=0, keepdims=True)
    prob_new = exp_scores / sum_exp

    # 3. Extract probability for the chosen class at each vertex
    pred_prob = prob_new[pred, np.arange(nv)]

    return pred, pred_prob


def fuse_folds(
    vertices: np.ndarray,
    faces: np.ndarray,
    fold_scores_list: Sequence[np.ndarray],
    *,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Refine and fuse class predictions across one or more model folds.

    Parameters
    ----------
    vertices : np.ndarray
        Mesh vertices, shape ``(nV, 3)``.
    faces : np.ndarray
        Mesh faces, shape ``(nF, 3)``.
    fold_scores_list : sequence of np.ndarray
        List of raw score arrays ``(nclass, nV)`` from each fold.
    labels : np.ndarray, optional
        1D array mapping 0-based class indices ``0 .. nclass-1`` to protocol integer IDs.
        If provided, the returned array contains the protocol IDs; otherwise 0-based indices.

    Returns
    -------
    pred_labels : np.ndarray
        1D int64 array of predicted labels for each vertex, shape ``(nV,)``.
    """
    if not fold_scores_list:
        raise ValueError("fold_scores_list cannot be empty")

    nfold = len(fold_scores_list)
    nclass, nv = fold_scores_list[0].shape

    pred_cand = np.zeros((nfold, nv), dtype=np.int64)
    pred_prob = np.zeros((nfold, nv), dtype=np.float64)

    for fold_idx, scores in enumerate(fold_scores_list):
        pred_cand[fold_idx], pred_prob[fold_idx] = refine_fold(vertices, faces, scores)

    if nfold == 1:
        # Single model (e.g. braincolor) skips fusion
        final_class = pred_cand[0]
    else:
        # Multi-model fusion (e.g. DKT31 5-fold)
        prob = np.zeros((nclass, nv), dtype=np.float64)
        for c in range(nclass):
            prob[c, :] = np.sum(pred_prob * (pred_cand == c), axis=0)

        sum_prob = np.sum(prob, axis=0, keepdims=True)
        # Avoid division by zero
        zero_mask = sum_prob == 0
        if np.any(zero_mask):
            sum_prob[zero_mask] = 1.0
        prob = prob / sum_prob

        # Log-likelihood with eps
        eps = np.finfo(np.float64).eps
        log_prob = np.log(prob + eps).astype(np.float32)

        final_class = gcut(vertices, faces, log_prob)
        final_class = roi_correction(final_class, vertices, faces, log_prob)

    if labels is not None:
        labels_arr = np.asarray(labels, dtype=np.int64)
        return labels_arr[final_class]
    return final_class
