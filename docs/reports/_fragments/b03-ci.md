## B03 task 4-5: CI macOS 3.10 job and cross-platform policy

### macOS + Python 3.10 job: not a real failure
- Run 39465318691 (https://github.com/BennettLandman/pyBiscuit/actions/runs/39465318691): job `py3.10 macos-latest`
  (id 118775077449) is `cancelled` with zero steps executed (started 17:21:11Z, ended
  17:36:13Z, no log exists). It never got a runner; it was cancelled at the 15 min run
  boundary/fail-fast, not by a test or install error. `py3.13 macos-latest` in the same run
  passed.
- Newest run 39469312755: macOS jobs were still queued when checked (3m47s), so no result yet.
- Resolved versions per job (pip install logs): py3.13 on all OSes: numpy 2.5.4, scipy 1.18.1,
  numba 0.68.0, llvmlite 0.50.0, nibabel 5.4.2. py3.10 (ubuntu, windows): numpy 2.2.6,
  scipy 1.15.3, numba 0.68.0, nibabel 5.4.2. py3.10 macOS has no install log to compare, but
  there is no evidence of a version-range issue (3.10 resolves the same as other 3.10 jobs).
  Note the numpy 2.2 vs 2.5 split between 3.10 and 3.13: a platform-independent difference.
- No change to pyproject.toml dependencies. `fail-fast: false` is already set in
  .github/workflows/ci.yml (kept). Numba caching not added (minimal change).
- Action: re-check py3.10 macOS in run 39469312755 once it finishes; if it fails there,
  inspect its log (expected to be the same exact-equality failures as the other jobs).

### Cross-platform policy (docs/validation-tolerances.md, PROPOSED)
E means within tolerance with ties explained on any platform; bit-exactness is diagnostic
(`pytest -m bitexact`, oracle platform only); acceptance reference is the x86-64 Linux
image; macOS arm64 oracle builds use -ffp-contract=off but are provisional; argmin/closest-face
mismatches accepted only if recomputed margin <= eps with tie count reported; candidate lists
compared as sets with eps-near-bound exceptions; default float32 rel 1e-5 / float64 rel 1e-10 with
abs scaled by golden magnitude; no tolerance loosened beyond the document; fixed-order
accumulation required where float32 rounding drives a discrete decision.
