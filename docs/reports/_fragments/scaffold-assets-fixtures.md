# Fragment: B0.2 CI, B0.3 asset registry, B0.4 fixtures, licenses

## Files
- `.github/workflows/ci.yml` - ruff check + pytest, {3.10, 3.13} x {ubuntu, macos, windows}, `pip install -e .[dev]`. YAML parsed OK with PyYAML (cannot run Actions locally). `reference` tests auto-skip via tests/conftest.py (no `reference/` in CI).
- `src/pybiscuit/assets.py`, `src/pybiscuit/data/{__init__.py,asset_registry.json}`, `scripts/build_asset_registry.py`; `pyproject.toml` package-data (`pybiscuit.data = ["*.json"]`). API: `load_registry()`, `status_summary()`, `verify(root)`, `cache_dir()` (`$PYBISCUIT_ASSETS` or `~/.cache/pybiscuit/assets`), `sha256_file()`. No download code.
- `tests/test_assets.py` (5 tests; 1 is `@pytest.mark.reference`: file set equals registry, sizes, all SHA-256), `tests/fixtures/{__init__,meshes}.py`, `tests/test_fixtures.py` (25 tests incl. parametrised). Combined 30 tests pass; reference test ran and passed locally.
- `LICENSE` (MIT, "pyBiscuit contributors", 2026), `LICENSES/{BISCUIT,CMorph,HSD,CurveExtraction,LocalGyrificationIndex,SphericalRemesh,SphericalLabeling,MeshLib}.txt` (verbatim copies), `LICENSES/UGSCNN-MeshConv.txt` (placeholder), `LICENSES/README.md` (index + not-included list).

## Registry
76 files, 1,259,450,198 bytes (reg 201M, parc 983M, sphere 18M); JSON is ~11 KB. Paths are relative to `/opt/biscuit` (e.g. `reg/mb101/lh.white.vtk`).

## Fixture hashes (`geometry_hash`: vertices -> `rint(v*1e9)` int64 LE, faces int32 LE, SHA-256)
| generator | sha256 |
|---|---|
| uv_sphere(8,16) | e8add2851dee170432d75169b39bdea355e0181173f726b551aabb930a3fca77 |
| ellipsoid(3,2,1,12,24) | 7eaaa40290a9885f90b126e6b9941ee73e289c058d4eea761e1ec38fb388fea9 |
| icosphere(0) | 47d05533271ff80f1013ab58306c11204e493aa614c07620ddcc889d55811d17 |
| icosphere(3) | 918dff3f719be3f13e9ec5a8a5fedb47c247eeb37998c9216687032098e8330c |
| icosphere(5) | 70b8fa3a73579410680cee8fafbaf2f7190a9ac9b5d8e8749088e9b266ded00b |
| folded_sphere(0.1,4,2,3) | f925ba9f161955586bd1c890ac54c627a380bb7f9d7bb6cff31eeb2f2425ac37 |
| cut_sphere(3,0.5) | 23c3d501031d1af9c3f01e46b937ace71ffa9a2af961ae0f43912b4490d5e47f |
| perturb(icosphere(2),0.01,seed=7) | 0e873076fea7b0084e0d3deda539713061f9c6278a4cc82c92be803a2e02d451 |

Only verified on this machine (macOS arm64, scipy 1.18.1); cross-platform stability relies on the 1e-9 quantisation (a value within ~1e-16 of a rounding boundary could in principle flip; unobserved). Real SH convention is documented in `meshes.py` (scipy `sph_harm_y`, falls back to `sph_harm`).

## Surprises / notes
- `tests/` must NOT be a package: other tests use `from conftest import ...`; mine do the same and import `fixtures` as a top-level package.
- UGSCNN MeshConv has no LICENSE text in `reference/` (SphericalLabeling/meshcnn has no header); placeholder only, verbatim MIT text to be copied from upstream.
- Existing LICENSES/ was an empty dir. The `ruff check --fix .` I ran repo-wide applied 2 auto-fixes (likely only import ordering) - possibly in other agents' files.
- At the time of my last run: `ruff check .` had 1 error in another agent's `src/pybiscuit/mesh/topology.py` (B905 zip strict) and `tests/test_operators.py::test_rejects_codecs_encode_misuse` failed (not mine).
