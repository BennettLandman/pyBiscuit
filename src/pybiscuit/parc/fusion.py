"""DKT31 five-model fusion -- interface only, rule pending B7.1 (plan row B7.3).

``dkt31`` ships five checkpoints per hemisphere (``?h.pth``, ``?h_{2..5}.pth``).  ``test.py``
runs one checkpoint per call and has **no** fusion step; combining the five outputs happens in the
encrypted MATLAB orchestrator (``parc.m``).  Inventing a rule (mean of scores, mean of softmax
probabilities, majority vote ...) would be a guess, so none is provided: callers must pass the rule
once it is recovered from the MATLAB source (Q7) or from instrumented image runs (B1).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["FusionRule", "fuse", "PENDING"]

FusionRule = Callable[[Sequence[np.ndarray]], np.ndarray]

PENDING = "DKT31 fusion rule pending B7.1 (parc.m is encrypted; see docs/open-questions.md Q7)"


def fuse(outputs: Sequence[np.ndarray], rule: FusionRule | None = None) -> np.ndarray:
    """Combine per-model outputs ``(C, nv)`` with an explicit ``rule``.

    Raises ``NotImplementedError`` when no rule is given.
    """
    if rule is None:
        raise NotImplementedError(PENDING)
    arrs = [np.asarray(o) for o in outputs]
    if len({a.shape for a in arrs}) != 1:
        raise ValueError("all model outputs must have the same shape")
    return rule(arrs)
