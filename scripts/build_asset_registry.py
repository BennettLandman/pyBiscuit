#!/usr/bin/env python3
"""Regenerate src/pybiscuit/data/asset_registry.json from the extracted reference image.

Lists every file under <root>/{reg,parc,sphere} with relative POSIX path, size
and SHA-256 (streamed). Default root: ../reference/sif_FC_v2.2/root/opt/biscuit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT.parent / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit"
OUT = REPO_ROOT / "src" / "pybiscuit" / "data" / "asset_registry.json"
SUBDIRS = ("reg", "parc", "sphere")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def build(root: Path) -> dict:
    files = []
    for sub in SUBDIRS:
        for p in sorted((root / sub).rglob("*")):
            if p.is_file():
                rel = p.relative_to(root).as_posix()
                files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256_file(p)})
    files.sort(key=lambda e: e["path"])
    return {
        "schema_version": 1,
        "source": "biscuit_FC_v2.2.sif:/opt/biscuit",
        "subdirs": list(SUBDIRS),
        "files": files,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    data = build(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    total = sum(f["size"] for f in data["files"])
    print(f"wrote {args.out}: {len(data['files'])} files, {total} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
