#!/usr/bin/env python3
"""Extract readable reference material from a BISCUIT Singularity image.

Writes, under --out:
  sif_<tag>/root/opt, sif_<tag>/root/.singularity.d   (via unsquashfs)
  ctf_plain/                                         (plaintext members of the MATLAB CTF)
and refreshes docs/reference/ctf-manifest.tsv in this repository.

Requires `unsquashfs` (squashfs-tools). Does not execute anything from the image.
"""

from __future__ import annotations

import argparse
import mmap
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "docs" / "reference" / "ctf-manifest.tsv"
CTF_PREFIXES = ("opt/BISCUIT/", "biscuit/")
ENCRYPTED_MAGIC = b"V1MCC"


def squashfs_offset(sif: Path) -> int:
    with sif.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as data:
        offset = data.find(b"hsqs")
    if offset < 0:
        raise SystemExit(f"No squashfs partition found in {sif}")
    return offset


def unsquash(sif: Path, offset: int, dest: Path) -> None:
    if shutil.which("unsquashfs") is None:
        raise SystemExit("unsquashfs not found; install squashfs-tools")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["unsquashfs", "-o", str(offset), "-d", str(dest), "-no-xattrs", "-f",
         str(sif), "/opt", "/.singularity.d"],
        check=True, stdout=subprocess.DEVNULL,
    )


def safe_member_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name.replace("opt/BISCUIT/", "", 1))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    return root.joinpath(*relative.parts)


def extract_ctf(executable: Path, out: Path) -> list[tuple[str, int, bool]]:
    rows = []
    with zipfile.ZipFile(executable) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.startswith(CTF_PREFIXES):
                continue
            payload = archive.read(info.filename)
            encrypted = payload.startswith(ENCRYPTED_MAGIC)
            rows.append((info.filename, info.file_size, encrypted))
            if not encrypted:
                target = safe_member_path(out, info.filename)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
    return rows


def write_manifest(rows: list[tuple[str, int, bool]]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w") as handle:
        handle.write("path\tbytes\tencrypted\n")
        for name, size, encrypted in rows:
            handle.write(f"{name}\t{size}\t{'yes' if encrypted else 'no'}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sif", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="reference directory (outside git)")
    parser.add_argument("--tag", help="name for sif_<tag>/ (default: from filename)")
    args = parser.parse_args(argv)

    tag = args.tag or args.sif.stem.replace("biscuit_", "")
    root = args.out / f"sif_{tag}" / "root"
    offset = squashfs_offset(args.sif)
    print(f"squashfs offset {offset}; extracting to {root}")
    unsquash(args.sif, offset, root)

    executable = root / "opt" / "biscuit" / "biscuit"
    rows = extract_ctf(executable, args.out / "ctf_plain")
    write_manifest(rows)
    encrypted = sum(1 for row in rows if row[2])
    print(f"CTF: {len(rows)} BISCUIT members, {encrypted} encrypted, {len(rows) - encrypted} plaintext")
    print(f"manifest: {MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
