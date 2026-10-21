#!/usr/bin/env bash
# Create the PATH-shim directory for command tracing (plan B1.2).
#   install_shims.sh [--dry-run] SHIM_DIR [TOOL...]
# Default tools: every sub-tool of the BISCUIT FC v2.2 image (see below).
set -euo pipefail

dry=0
if [ "${1:-}" = "--dry-run" ]; then
    dry=1
    shift
fi
[ $# -ge 1 ] || { echo "usage: $0 [--dry-run] SHIM_DIR [TOOL...]" >&2; exit 2; }
dir=$1
shift
# /opt/biscuit/bin and /opt/cmorph/bin of biscuit_FC_v2.2.sif
default_tools=(MeshContour MeshDeform MeshProperty ShapeComplexityIndex mri_convert mri_normalize
    mris_inflate mris_make_surfaces mris_sphere CurveExtraction Gyrification HSD OuterHull
    SphericalRemesh klaplace lgi)
tools=("$@")
[ ${#tools[@]} -gt 0 ] || tools=("${default_tools[@]}")

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
run() { if [ $dry -eq 1 ]; then echo "+ $*"; else "$@"; fi; }
run mkdir -p "$dir"
run cp "$here/trace_shim.sh" "$dir/trace_shim.sh"
run chmod 755 "$dir/trace_shim.sh"
for t in "${tools[@]}"; do
    run ln -sf trace_shim.sh "$dir/$t"
done
