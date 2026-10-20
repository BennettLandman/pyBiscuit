"""Shared golden-comparison helpers (cross-platform policy, brief 03).

E-class means "within tolerance, with ties explained", on any platform.  Bit-exact
comparison is diagnostic only (``pytest -m bitexact``, skipped by default and in CI).

Tolerances come from docs/validation-tolerances.md: float32 paths ``rel=1e-5``, float64
paths ``rel=1e-10``.  Because relative error is meaningless next to zero, ``abs_`` defaults
to ``rel`` times the largest magnitude in the golden array (a "robust range" scale).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

REL32 = 1e-5
REL64 = 1e-10


def _check_close(a, b, rel, abs_, label):
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    if abs_ is None:
        finite = np.abs(b[np.isfinite(b)]) if b.size else b
        abs_ = rel * (float(finite.max()) if finite.size else 0.0)
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    nan_a, nan_b = np.isnan(a64), np.isnan(b64)
    assert np.array_equal(nan_a, nan_b), f"{label}: NaN pattern differs"
    ok = nan_a | np.isclose(a64, b64, rtol=rel, atol=abs_, equal_nan=True)
    if not ok.all():
        bad = np.flatnonzero(~ok.ravel())
        d = np.abs(a64 - b64).ravel()[bad]
        raise AssertionError(
            f"{label}: {bad.size}/{ok.size} values outside rel={rel:g}, abs={abs_:g}; "
            f"max |diff|={d.max():.3e} at flat index {bad[d.argmax()]}"
        )


def assert_close_float32(a, b, rel: float = REL32, abs_: float | None = None, label="float32"):
    """float32-path comparison (default rel=1e-5, abs scaled by golden's magnitude)."""
    _check_close(a, b, rel, abs_, label)


def assert_close_float64(a, b, rel: float = REL64, abs_: float | None = None, label="float64"):
    """float64-path comparison (default rel=1e-10)."""
    _check_close(a, b, rel, abs_, label)


def assert_same_choice_modulo_ties(
    chosen,
    golden,
    margin: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | np.ndarray,
    eps: float,
    label: str = "choice",
) -> int:
    """argmin / closest-face style results: a mismatch is allowed only when it is a tie.

    ``margin(idx, chosen[idx], golden[idx])`` must RECOMPUTE, for the mismatched query
    indices, the score gap |score(chosen) - score(golden)| (e.g. the point-to-triangle
    distance difference between the two faces).  A mismatch with gap <= eps counts as a tie;
    any untied mismatch fails.  ``margin`` may also be a precomputed array over all queries.
    Returns the tie count.
    """
    chosen = np.asarray(chosen)
    golden = np.asarray(golden)
    assert chosen.shape == golden.shape, f"{label}: shape {chosen.shape} != {golden.shape}"
    bad = np.flatnonzero(chosen != golden)
    if bad.size == 0:
        return 0
    gap = (
        np.asarray(margin)[bad]
        if not callable(margin)
        else np.asarray(margin(bad, chosen[bad], golden[bad]))
    )
    untied = bad[~(np.abs(gap) <= eps)]
    assert untied.size == 0, (
        f"{label}: {untied.size} untied mismatches (of {bad.size}), first idx "
        f"{untied[:5].tolist()}, gaps {np.abs(gap)[~(np.abs(gap) <= eps)][:5].tolist()}"
    )
    return int(bad.size)


def assert_candidate_sets_close(
    got: Sequence,
    expected: Sequence,
    near: Callable[[int, int], bool],
    label: str = "candidates",
) -> int:
    """Candidate lists (AABB search) compared as sets.

    ``near(query_index, face_index)`` returns True if that face's box bound is within eps of
    the query (so its inclusion is legitimately platform-dependent).  Symmetric-difference
    members for which ``near`` is False fail.  Returns the number of lists that differed.
    """
    assert len(got) == len(expected), f"{label}: {len(got)} != {len(expected)} lists"
    n_diff = 0
    for i, (g, e) in enumerate(zip(got, expected, strict=True)):
        sg, se = set(np.asarray(g).tolist()), set(np.asarray(e).tolist())
        if sg == se:
            continue
        n_diff += 1
        extra = [f for f in sg ^ se if not near(i, f)]
        assert not extra, f"{label}: query {i}: faces {extra[:5]} differ and are not near a bound"
    return n_diff
