#!/usr/bin/env bash
# Runs INSIDE the image (copied to <out>.harness/ and bound at /harness by run_full.sh).
# run_biscuit with its two cleanup lines removed, so $OUT/.tmp (intermediates) survives.
set -eo pipefail
# shellcheck disable=SC2016 # literal $TMP / $CONF are what sed must match
sed -e '/^rm -rf \$TMP$/d' -e '/^rm \$CONF$/d' /opt/biscuit/run_biscuit >/harness/run_biscuit.keep
chmod +x /harness/run_biscuit.keep
if [ "${TRACE:-1}" = 1 ]; then export PATH=/harness/shims:$PATH; fi
args=(--reg "$REG" --surf_recon "$SURF_RECON")
if [ -n "${THREAD:-}" ]; then args+=(--thread "$THREAD"); fi
# shellcheck disable=SC2206 # EXTRA is a deliberate word-split option string
args+=(${EXTRA:-})
if [ "${STRACE:-0}" = 1 ]; then
    exec strace -f -e trace=execve -o /harness/execve.strace \
        /harness/run_biscuit.keep "${args[@]}" /INPUTS /OUTPUTS
fi
exec /harness/run_biscuit.keep "${args[@]}" /INPUTS /OUTPUTS
