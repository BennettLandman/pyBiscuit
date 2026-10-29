#!/usr/bin/env bash
# Build the image-job bundle that someone with a Linux x86-64 + Apptainer host runs for us
# (instructions: scripts/reference/bundle/README.md, shipped as the bundle's README.md).
#
#   make_bundle.sh [--out DIR] [--macruise MACRUISE_OUTPUT_DIR] [--freesurfer-root DIR]
#                  [--subjects "0050952 0050961 0050971"] [--allow-dirty]
#
# Writes DIR/pybiscuit-image-jobs-<date>-<commit>/ and the .tar.gz of it. The bundle holds a
# `git archive` of HEAD (so commit first), the driver, and staged inputs: the MaCRUISE v3 sample
# (subject-derived) and the FreeSurfer files of the open-examples subjects. It never holds the
# image or a licence.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)

out=$HOME/dev/biscuit/bundles
macruise=$HOME/dev/mipav/sample-macruise/output
fsroot=${PYBISCUIT_OPEN_EXAMPLES:-$HOME/dev/biscuit/open-examples}
subjects="0050952 0050961 0050971"
dirty=0
while [ $# -gt 0 ]; do
    case $1 in
        --out) out=$2; shift 2 ;;
        --macruise) macruise=$2; shift 2 ;;
        --freesurfer-root) fsroot=$2; shift 2 ;;
        --subjects) subjects=$2; shift 2 ;;
        --allow-dirty) dirty=1; shift ;;
        -h | --help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done

cd "$repo"
if [ "$dirty" = 0 ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "error: uncommitted changes; the bundle is built from HEAD (commit, or pass --allow-dirty)" >&2
    exit 1
fi
commit=$(git rev-parse --short HEAD)
name=pybiscuit-image-jobs-$(date +%Y%m%d)-$commit
dst=$out/$name
[ ! -e "$dst" ] || { echo "error: $dst exists" >&2; exit 1; }
mkdir -p "$dst/pyBiscuit" "$dst/inputs/freesurfer"

git archive HEAD | tar -x -C "$dst/pyBiscuit"
cp "$here/bundle/README.md" "$here/bundle/run_jobs.sh" "$dst/"
chmod +x "$dst/run_jobs.sh"
printf 'pyBiscuit %s (%s), built %s\n' "$commit" "$(git log -1 --format=%cs)" "$(date -u +%Y-%m-%dT%H:%MZ)" \
    >"$dst/BUNDLE_VERSION"

"$here/stage_macruise_input.sh" "$macruise" "$dst/inputs/macruise/sample"
for s in $subjects; do
    "$here/stage_freesurfer_input.sh" "$fsroot/$s" "$dst/inputs/freesurfer/$s"
done

(
    cd "$dst"
    find . -type f ! -name BUNDLE.sha256 | sed 's|^\./||' | LC_ALL=C sort | while IFS= read -r f; do
        if command -v sha256sum >/dev/null 2>&1; then sha256sum "$f"; else shasum -a 256 "$f"; fi
    done >BUNDLE.sha256
)
tar -czf "$dst.tar.gz" -C "$out" "$name"
echo "bundle: $dst.tar.gz ($(du -h "$dst.tar.gz" | cut -f1))"
if command -v sha256sum >/dev/null 2>&1; then sha256sum "$dst.tar.gz"; else shasum -a 256 "$dst.tar.gz"; fi
