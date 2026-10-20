# Upstream oracles (macOS arm64)

Oracles are the upstream C++ tools built from `reference/repos/` at pinned commits into
`~/dev/biscuit/oracles/<tool>/` (outside git). Details, patches and per-tool status:
[docs/ports/oracles.md](../../docs/ports/oracles.md). Tests that need them must skip when
`~/dev/biscuit/oracles` is absent. Committed goldens (`tests/data/golden/<tool>/<case>/`)
never need the binaries; regenerate with `scripts/oracles/make_golden.py <tool> <case>`
(`--list` shows registered tools; add a tool by registering a function, see its docstring).

## MeshLib (O1a)

Build: `scripts/oracles/build_meshlib.sh` (plain clang++, `-std=c++11 -O2 -DNDEBUG`, all
sources of upstream CMakeLists incl. Geodesic; pinned `dcef807`; one build-only patch).
`oracles/meshlib/READY` exists when the library is usable.

- Include dirs: `O=$HOME/dev/biscuit/oracles/meshlib`; `-I$O/include -I$O/include/Util`
  (add `-I$O/include/Geodesic` and `-I$O/include/Geodesic/gw/{gw_core,gw_geodesic,gw_maths}`
  only if you use geodesics). Headers are included as `"Mesh.h"`, `"SurfaceUtil.h"`,
  `"SphericalHarmonics.h"`, `"AABB_Sphere.h"`, `"Geom.h"`.
- Library: `$O/lib/libMesh.a` (static). No LAPACK/Accelerate/OpenMP needed.
- Compile line:
  `clang++ -std=c++11 -O2 -I$O/include -I$O/include/Util harness.cpp $O/lib/libMesh.a -o harness`
- Vertices are `float` inside MeshLib; use `Mesh::openFile(path)` for VTK (ASCII or binary).

## SphericalRemesh (O1a)

Build: `scripts/oracles/build_spherical_remesh.sh` -> `oracles/spherical-remesh/bin/SphericalRemesh`
(pinned `b677c39`; needs CLI11 v1.9.1 single header, cached at `oracles/_dl/CLI11.hpp`).
Smoke command (inputs from `make_golden.sr_fixture`, cwd = fixture dir):

    SphericalRemesh -s sphere.vtk -i subject.vtk -r ref.vtk -p prop.txt \
        -o out.vtk --outputProperty outprop --bary bary.txt --quiet

Property files are plain text, one float per sphere (`-s`) vertex; the remeshed property is
written to `<outputProperty>.<name>.txt` where `<name>` comes from the property file name
(`prop.txt` -> `outprop.prop.txt`).

## O1b tools

Builds: `scripts/oracles/build_{hsd,curve_extraction,gyrification}.sh` (shared helpers in
`_common_o1b.sh`; they link O1a's `oracles/meshlib/lib/libMesh.a` when `oracles/meshlib/READY`
exists, else compile the same dcef807 sources into `oracles/_deps/meshlib`; `MESHLIB_LIB=` /
`O1B_OWN_MESHLIB=1` override). CLI11 headers (v1.9.1 HSD, v1.9.0 others) are cached in
`oracles/_deps/cli11-*/`. Binaries: `oracles/hsd/bin/HSD`, `oracles/curve_extraction/bin/CurveExtraction`,
`oracles/gyrification/bin/Gyrification`. Build-only shims: `patches/hsd-shim/` (lapacke.h/cblas.h
-> Accelerate), `patches/curve-gyr-omp-shim/omp.h` (serial stub). Run from a fixture dir:

    HSD -s s1.sphere.vtk s2.sphere.vtk -p s1.curv.txt s2.curv.txt -o s1.reg.vtk s2.reg.vtk -d 8 --icosahedron 4 --maxIter 3
    CurveExtraction -i surf.vtk -o out --sulcus --gyrus --cart --bary --interm --sulcalSeed -0.01 --gyralSeed 0.01
    Gyrification -i surf.vtk --outer hull.vtk -g out.gcurve -s out.scurve -o lgi   # -> lgi.lgi.map.<area>.txt

HSD needs >= 2 subjects (`-s`/`-p`/`-o` equal counts); properties are one float per line.
Smoke tests: `tests/test_oracle_smoke_{hsd,curve,gyr}.py`.
