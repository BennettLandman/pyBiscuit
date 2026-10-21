"""Inference with a converted checkpoint and output resampling (plan rows B7.2, B7.3).

Mirrors ``test.py``: features ``(in_ch, nv)`` on the level-``max_level`` icosphere go through the
network in eval mode, the raw per-class scores (``test.py`` exports the output of the last
``MeshConv``; there is no softmax) are written with ``np.savetxt(fmt='%f')`` and resampled from
the icosahedral sphere to the native sphere with ``SphericalRemesh -s <isphere> -r <osphere>``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch

from ..sphere import remesh
from .checkpoint import load_safetensors
from .unet import OperatorCache, SphericalUNet, infer_architecture

__all__ = ["ParcModel", "resample_to_native", "text_roundtrip"]


class ParcModel:
    """A :class:`SphericalUNet` loaded from a converted safetensors checkpoint."""

    def __init__(self, net: SphericalUNet, meta: dict | None = None):
        self.net = net.eval()
        self.meta = meta or {}

    @classmethod
    def from_safetensors(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        operators: Callable[[int], dict] | OperatorCache | None = None,
    ) -> ParcModel:
        tensors, meta = load_safetensors(path, expected_sha256=expected_sha256)
        arch = meta.get("arch") or infer_architecture(tensors)
        net = SphericalUNet(operators=operators, **arch)
        return cls(_load_strict(net, tensors), meta)

    @property
    def in_ch(self) -> int:
        return self.net.in_ch

    @property
    def out_ch(self) -> int:
        return self.net.out_ch

    @property
    def nv(self) -> int:
        return 10 * 4**self.net.max_level + 2

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Raw class scores ``(out_ch, nv)`` float32 for features ``(>= in_ch, >= nv)``.

        As in ``loader_individual.py`` only the first ``in_ch`` channels and the first ``nv``
        vertices are used (icosphere vertex order is hierarchical, so a finer file is truncated).
        """
        f = np.asarray(features)
        if f.ndim != 2 or f.shape[0] < self.in_ch or f.shape[1] < self.nv:
            raise ValueError(f"features {f.shape}: need at least ({self.in_ch}, {self.nv})")
        x = torch.from_numpy(np.ascontiguousarray(f[: self.in_ch, : self.nv], dtype=np.float32))
        return self.net(x[None])[0].numpy()


def _load_strict(net: SphericalUNet, tensors: dict) -> SphericalUNet:
    """Load canonical tensors; the only tolerated absentees are BN ``num_batches_tracked``."""
    res = net.load_state_dict(tensors, strict=False)
    missing = [k for k in res.missing_keys if not k.endswith("num_batches_tracked")]
    if missing or res.unexpected_keys:
        raise ValueError(
            f"checkpoint/model mismatch: missing {missing[:5]}, "
            f"unexpected {res.unexpected_keys[:5]}"
        )
    return net


def text_roundtrip(scores: np.ndarray) -> np.ndarray:
    """What ``np.savetxt(fmt='%f')`` followed by SphericalRemesh's ``%f`` read does to scores.

    Each value is printed with six decimals and parsed back into float32.
    """
    flat = np.asarray(scores, dtype=np.float32).ravel()
    txt = np.array([f"{float(x):f}" for x in flat], dtype=np.float64)
    return txt.astype(np.float32).reshape(np.shape(scores))


def resample_to_native(
    scores: np.ndarray,
    isphere,
    osphere,
    *,
    quantize: bool = True,
) -> np.ndarray:
    """Per-class resampling of ``(C, nv)`` scores from ``isphere`` to the native ``osphere``.

    Equivalent of ``SphericalRemesh -p <class files> -r osphere -s isphere --outputProperty ...``
    (barycentric interpolation, :func:`pybiscuit.sphere.remesh.resample`).  ``isphere`` must have
    exactly ``nv`` vertices (the level the network ran on).  With ``quantize`` the six-decimal text
    round trip of ``test.py`` is reproduced first.
    """
    scores = np.asarray(scores)
    if quantize:
        scores = text_roundtrip(scores)
    rows: Sequence[np.ndarray] = [scores[i] for i in range(scores.shape[0])]
    out = remesh.resample(isphere, osphere, rows)
    return np.stack(out).astype(np.float32)
