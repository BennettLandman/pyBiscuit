#!/usr/bin/env python
"""Convert the 12 legacy parcellation checkpoints to safetensors (plan row B7.2).

    python scripts/convert_parc_checkpoints.py [--src DIR] [--out DIR] [--write-registry]

``--src``  directory laid out like ``/opt/biscuit/parc`` (default: the extracted FC v2.2 image).
``--out``  destination (default: ``$PYBISCUIT_ASSETS`` or ``~/.cache/pybiscuit/assets``); files
           land at ``<out>/parc/<protocol>/<name>.safetensors``.
``--write-registry``  rewrite ``src/pybiscuit/data/parc_converted_registry.json`` (hashes only).

Each source file is verified against the asset registry SHA-256, audited with ``pickletools`` and
loaded with ``torch.load(weights_only=True)``; see ``pybiscuit.parc.checkpoint``.  The converted
weights are derived from third-party models whose redistribution is unresolved (Q6/Q7): keep them
out of git and out of any release.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pybiscuit.assets import cache_dir
from pybiscuit.parc.checkpoint import convert_checkpoint

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = Path.home() / "dev/biscuit/reference/sif_FC_v2.2/root/opt/biscuit/parc"
REGISTRY = REPO / "src/pybiscuit/data/parc_converted_registry.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--write-registry", action="store_true")
    a = ap.parse_args(argv)
    out = a.out or cache_dir()
    srcs = sorted(a.src.glob("*/?h*.pth"))
    if len(srcs) != 12:
        print(f"expected 12 checkpoints under {a.src}, found {len(srcs)}", file=sys.stderr)
        return 1
    entries = []
    for s in srcs:
        t = time.perf_counter()
        dst = out / "parc" / s.parent.name / (s.stem + ".safetensors")
        e = convert_checkpoint(s, dst)
        entries.append(e)
        print(f"{e['path']}: {e['size']:,} B  sha256 {e['sha256'][:16]}  "
              f"{e['tensors']} tensors  NS/EW diff {e['ns_ew_max_abs_diff']:.2e}  "
              f"{time.perf_counter() - t:.1f}s")  # fmt: skip
    if a.write_registry:
        torch_versions = sorted({e.pop("torch") for e in entries})
        doc = {
            "schema_version": 1,
            "format": "pybiscuit-parc-unet/1",
            "note": "hashes of converted checkpoints only; the weights are not redistributable "
            "(open-questions Q6/Q7)",
            "converted_with_torch": torch_versions,
            "files": entries,
        }
        REGISTRY.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        print(f"wrote {REGISTRY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
