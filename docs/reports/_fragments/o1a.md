# O1a fragment: MeshLib + SphericalRemesh oracles

- Commits verified: MeshLib `dcef807`, SphericalRemesh `b677c39`. Toolchain Apple clang 21.0.0, arm64, `-std=c++11 -O2`.
- MeshLib builds in a few seconds, one build-only patch (`<cstddef>` in `Util/Geom.h`). No LAPACK is used by MeshLib or SphericalRemesh, so `-framework Accelerate` is not needed. `READY` marker at `oracles/meshlib/READY`.
- SphericalRemesh needs CLI11 v1.9.1 (CMake downloads it at configure time); fetched once from the GitHub release and cached in `oracles/_dl/` (needs network once; sandbox allowed github.com).
- Smoke command: `SphericalRemesh -s sphere.vtk -i subject.vtk -r ref.vtk -p prop.txt -o out.vtk --outputProperty outprop --bary bary.txt --quiet`. Fixtures: `make_golden.sr_fixture` (icosphere(5) source, icosphere(4)/(6) reference, radial-bump subject, seeded property seed 20261020). 5->6 runs in ~0.65 s.
- Surprises: property output name is derived from the property filename (`outprop.prop.txt`); bary file rows are `id w0 w1 w2` (`-0.000000` appears); text outputs are 6 decimals so goldens compare at ~1e-6. `up5to6` outputs are >5 MB so only `down5to4` and `up3to4` goldens are committed (about 300 KB each); the 5->6 case runs live in the test.
- No `-i` is required for property remeshing; `-c` (HSD coefficients) not exercised in the smoke test.
- FP contraction/FMA: not disabled; compare to x86-64 SIF may differ in last bits.
