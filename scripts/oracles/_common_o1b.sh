# Shared helpers for the O1b oracle builds (sourced by build_{hsd,curve_extraction,gyrification}.sh).
# Compiles the upstream MeshLib (dcef807) sources into a static lib unless O1a's lib is ready.
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BISCUIT=${BISCUIT:-$HOME/dev/biscuit}
REF=${REF:-$BISCUIT/reference/repos}
ML=$REF/MeshLib
ORACLES=${ORACLES:-$BISCUIT/oracles}
DEPS=$ORACLES/_deps
CXX=${CXX:-/usr/bin/clang++}
export SDKROOT=${SDKROOT:-/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk}
ML_INC="-I$ML/include -I$ML/include/Geodesic -I$ML/include/Geodesic/gw -I$ML/include/Geodesic/gw/gw_core -I$ML/include/Geodesic/gw/gw_geodesic -I$ML/include/Geodesic/gw/gw_maths -I$ML/include/Util"
# -include cstddef: build-only (old MeshLib headers use size_t/NULL without including it); no numeric effect.
CXXFLAGS_COMMON="-O2 -DNDEBUG -std=c++11 -include cstddef"

fetch_cli11() {  # $1 = version; vendored upstream release header (same URL as upstream CMake file(DOWNLOAD))
  d=$DEPS/cli11-$1
  if [ ! -s "$d/CLI11.hpp" ]; then
    mkdir -p "$d"
    curl -sSL -o "$d/CLI11.hpp" "https://github.com/CLIUtils/CLI11/releases/download/v$1/CLI11.hpp"
  fi
  echo "$d"
}

build_meshlib() {  # builds $DEPS/meshlib/libMesh.a from reference sources (identical file list to MeshLib/CMakeLists.txt)
  out=$DEPS/meshlib
  [ -f "$out/libMesh.a" ] && { echo "$out/libMesh.a"; return; }
  mkdir -p "$out/obj"
  srcs="src/Mesh.cpp src/MeshIO.cpp src/MNIObjIO.cpp src/VtkIO.cpp
   src/Util/Geom.cpp src/Util/AABB.cpp src/Util/AABB_Sphere.cpp src/Util/Slicer.cpp src/Util/SphericalHarmonics.cpp src/Util/SurfaceUtil.cpp
   src/Geodesic/Geodesic.cpp src/Geodesic/GeodesicA.cpp src/Geodesic/GeodesicPath.cpp
   src/Geodesic/gw/gw_core/GW_Config.cpp src/Geodesic/gw/gw_core/GW_FaceIterator.cpp src/Geodesic/gw/gw_core/GW_SmartCounter.cpp
   src/Geodesic/gw/gw_core/GW_VertexIterator.cpp src/Geodesic/gw/gw_core/GW_Face.cpp src/Geodesic/gw/gw_core/GW_Mesh.cpp src/Geodesic/gw/gw_core/GW_Vertex.cpp
   src/Geodesic/gw/gw_geodesic/GW_GeodesicFace.cpp src/Geodesic/gw/gw_geodesic/GW_GeodesicMesh.cpp src/Geodesic/gw/gw_geodesic/GW_GeodesicMeshA.cpp
   src/Geodesic/gw/gw_geodesic/GW_GeodesicPath.cpp src/Geodesic/gw/gw_geodesic/GW_GeodesicPoint.cpp
   src/Geodesic/gw/gw_geodesic/GW_TriangularInterpolation_Cubic.cpp src/Geodesic/gw/gw_geodesic/GW_GeodesicVertex.cpp
   src/Geodesic/gw/gw_geodesic/GW_TriangularInterpolation_Linear.cpp src/Geodesic/gw/gw_geodesic/GW_TriangularInterpolation_Quadratic.cpp"
  for f in $srcs; do
    o="$out/obj/$(echo "$f" | tr / _).o"
    echo "$f $o"
  done | xargs -P 8 -n 2 sh -c '$0 -O2 -std=c++11 -include cstddef '"$ML_INC"' -c '"$ML"'/$1 -o $2' "$CXX" 2>&1 | grep -v warning || true
  rm -f "$out/libMesh.a"
  /Library/Developer/CommandLineTools/usr/bin/ar rcs "$out/libMesh.a" "$out"/obj/*.o
  echo "$out/libMesh.a"
}

# use_meshlib: sets LIB (+ ML_INC). Prefers O1a's oracles/meshlib (patched copy of dcef807) when READY exists;
# MESHLIB_LIB overrides; O1B_OWN_MESHLIB=1 forces our own build of the identical sources.
use_meshlib() {
  if [ -n "$MESHLIB_LIB" ]; then LIB=$MESHLIB_LIB
  elif [ -f "$ORACLES/meshlib/READY" ] && [ -z "$O1B_OWN_MESHLIB" ]; then
    M=$ORACLES/meshlib
    ML_INC="-I$M/include -I$M/include/Geodesic -I$M/include/Geodesic/gw -I$M/include/Geodesic/gw/gw_core -I$M/include/Geodesic/gw/gw_geodesic -I$M/include/Geodesic/gw/gw_maths -I$M/include/Util"
    LIB=$M/lib/libMesh.a
  else LIB=$(build_meshlib)
  fi
  echo "MeshLib: $LIB"
}
