#!/bin/bash
# Stage the image's original parcellation code as an oracle (plan row B7.2-oracle).
# Creates $ORACLES/parc-unet/{venv,stage}: a throwaway venv with current torch/numpy/scipy and a
# byte-for-byte copy of ctf_plain/SurfPostproc/common/parc (python sources, meshcnn/, mesh_files/
# pickles).  Nothing is edited; shims live in parc_unet_oracle.py.  SHA-256 of every staged file
# is written to $OUT/stage.sha256 so the copy can be audited.
set -euo pipefail
ROOT="${BISCUIT_ROOT:-$HOME/dev/biscuit}"
ORA="${ORACLES:-$ROOT/oracles}"
SRC="$ROOT/reference/ctf_plain/SurfPostproc/common/parc"
OUT="$ORA/parc-unet"
PY=${PYTHON:-python3}
[ -f "$SRC/test.py" ] || { echo "missing $SRC (run scripts/extract_reference.py)" >&2; exit 1; }
mkdir -p "$OUT"
if [ ! -x "$OUT/venv/bin/python" ]; then
  "$PY" -m venv "$OUT/venv"
  "$OUT/venv/bin/pip" install -q torch numpy scipy
fi
rm -rf "$OUT/stage"
mkdir -p "$OUT/stage"
cp -R "$SRC/." "$OUT/stage/"
(cd "$OUT/stage" && find . -type f | sort | xargs shasum -a 256) > "$OUT/stage.sha256"
"$OUT/venv/bin/python" -c "import torch, numpy, scipy; print('torch', torch.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)" > "$OUT/versions.txt"
cat "$OUT/versions.txt"
echo "OK: $OUT/stage"
