#!/usr/bin/env bash
# Stage a MaCRUISE v3 output folder into the flat input layout that
# biscuit_FC_v2.2.sif's biscuit.conf expects (as MASI production scripts do):
#   target_image_GMimg_innerSurf.asc      <- Surfaces_FreeView/target_image_GMimg_innerSurf.asc
#   target_image_GMimg_innerSurf_mni.asc  <- Surfaces_MNI/target_image_GMimg_innerSurf.asc
#   T1.nii.gz                             <- Raw/T1.nii.gz
#   SEG_refine.nii.gz                     <- SegRefine/SEG_refine.nii.gz
#
#   stage_macruise_input.sh MACRUISE_OUTPUT_DIR STAGED_INPUT_DIR
set -euo pipefail

if [ "$#" -ne 2 ]; then
    sed -n '2,10p' "${BASH_SOURCE[0]}"
    exit 2
fi
src=$1
dst=$2
mkdir -p "$dst"

stage() {
    [ -f "$src/$1" ] || { echo "missing: $src/$1" >&2; exit 1; }
    cp "$src/$1" "$dst/$2"
}
stage Surfaces_FreeView/target_image_GMimg_innerSurf.asc target_image_GMimg_innerSurf.asc
stage Surfaces_MNI/target_image_GMimg_innerSurf.asc target_image_GMimg_innerSurf_mni.asc
stage Raw/T1.nii.gz T1.nii.gz
stage SegRefine/SEG_refine.nii.gz SEG_refine.nii.gz

(cd "$dst" && sha256sum -- * > INPUTS.sha256 2>/dev/null || shasum -a 256 -- * > INPUTS.sha256)
echo "staged $(ls "$dst" | wc -l | tr -d ' ') files in $dst"
