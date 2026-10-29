#!/usr/bin/env bash
# Stage the FreeSurfer files that biscuit_FC_v2.2.sif reads on the FreeSurfer route
# (--surf_recon freesurfer; BISCUIT/GenericRecon/biscuit.m:148-181) into a clean subject directory:
#   surf/{lh,rh}.{white,sphere,inflated,pial,curv,sulc,inflated.H,thickness}
#   label/{lh,rh}.cortex.label
# Any BISCUIT products already in the source (surf/*.vtk, quant/, ...) are left behind, so the image
# recomputes everything.
#
#   stage_freesurfer_input.sh FREESURFER_SUBJECT_DIR STAGED_INPUT_DIR
set -euo pipefail

if [ "$#" -ne 2 ]; then
    sed -n '2,9p' "${BASH_SOURCE[0]}"
    exit 2
fi
src=$1
dst=$2
mkdir -p "$dst/surf" "$dst/label"

for h in lh rh; do
    for f in white sphere inflated pial curv sulc inflated.H thickness; do
        [ -f "$src/surf/$h.$f" ] || { echo "missing: $src/surf/$h.$f" >&2; exit 1; }
        cp "$src/surf/$h.$f" "$dst/surf/"
    done
    [ -f "$src/label/$h.cortex.label" ] || { echo "missing: $src/label/$h.cortex.label" >&2; exit 1; }
    cp "$src/label/$h.cortex.label" "$dst/label/"
done

(cd "$dst" && { sha256sum surf/* label/* 2>/dev/null || shasum -a 256 surf/* label/*; } > INPUTS.sha256)
echo "staged 18 files in $dst"
