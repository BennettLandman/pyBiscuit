# Agent brief 03 — Portable numerics (fix CI on PR #2)

**Written 2026-10-20.** Small, sequential, and blocking: briefs 04–07 start
only after this lands. Read [AGENTS.md](../../AGENTS.md) and the
[brief 02 report](../reports/geometry-remesh-2026-10-20.md).

## Problem

PR #2 passes locally (macOS arm64, 360 tests) but CI fails on 5 of 6 jobs.
Failures on Linux and Windows x86-64 (and macOS + Python 3.10) are
**exact-equality** checks against oracle dumps produced on arm64:

- `tests/test_curvature.py::test_matches_committed_oracle_dumps[*]`:
  `np.array_equal` on face/vertex normals (line ~53).
- `tests/test_aabb_sphere.py::test_golden[aabb_ico3_*]`: 2–4 face indices
  differ, and 22 candidate lists differ in `aabb_ico3_search`.

The goal is reproducing an **x86-64 Linux image**, so passing only on arm64
is the wrong target. NumPy's SIMD reductions, summation order, and libm
differ by platform; bit-identity across platforms is not a valid acceptance
criterion.

## Tasks

1. **Comparison policy.** Add `tests/_compare.py` with shared helpers, and use
   them everywhere goldens are compared:
   - `assert_close_float32(a, b, rel=1e-5, abs_=…)` and a float64 variant
     (`rel=1e-10`), with defaults taken from
     [validation-tolerances.md](../validation-tolerances.md);
   - `assert_same_choice_modulo_ties(chosen, golden, margins, eps)`: for argmin
     and closest-face style results, a mismatch is allowed only when the
     golden's runner-up is within `eps` of the winner. The test must
     recompute the margin and report the tie count. Untied mismatches fail.
   - Candidate lists (AABB search): compare as sets; allow differences only for
     boxes whose bounds are within `eps` of the query.
2. **Remove `np.array_equal` / `==` on floats** in golden comparisons
   (grep the tests). Keep bit-exact checks only as an opt-in local check
   (`pytest -m bitexact`, skipped by default and in CI) for regression hunting
   on the oracle platform.
3. **Find and pin nondeterminism in the code**, not just in the tests: wherever
   float32 rounding drives a discrete decision (AABB pruning, nearest face,
   ties), compute it in a defined order. Use explicit loops in Numba or
   `math.fsum`-style fixed-order sums, not `np.sum` over vectors that
   NumPy may SIMD-reduce. Document each such site in the relevant port note.
4. **macOS + Python 3.10 job:** find the cause (likely a different NumPy/SciPy
   version range on 3.10). Fix it or pin a lower bound in `pyproject.toml`
   with justification.
5. **Tolerance policy doc:** add a "Cross-platform policy" section to
   validation-tolerances.md, marked PROPOSED for owner approval: E-class means
   "within tolerance, with ties explained", on any platform. Bit-exactness is
   diagnostic only.
6. Re-run CI until it is green on all 6 jobs. Do not loosen a tolerance beyond
   validation-tolerances.md to get there. If a check still fails, report it
   with the measured difference.

## Out of scope

New algorithms. Changes to oracle builds (they stay as built; the
`-ffp-contract=off` rebuild of HSD/CurveExtraction/Gyrification is in briefs
04–07).

## Hand-back

Commit to `b2-b4-geometry-remesh` (PR #2). Update the PR #2 description with
the policy change. Add `docs/reports/portable-numerics-YYYY-MM-DD.md`
(per-test before/after, tie counts, root causes). Final message: CI status
with the run link, plus any check that needed a tolerance or tie argument.
