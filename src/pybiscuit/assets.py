"""Asset registry: relative path, size and SHA-256 of every template/model file.

Data only. There is deliberately no download code here; the registry pins what
the BISCUIT 2.2 image ships under ``/opt/biscuit/{reg,parc,sphere}`` so that
any directory of assets can be verified against it. Regenerate the committed
registry with ``scripts/build_asset_registry.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path

SCHEMA_VERSION = 1
REGISTRY_RESOURCE = "asset_registry.json"
CACHE_ENV = "PYBISCUIT_ASSETS"

__all__ = [
    "AssetEntry",
    "VerifyReport",
    "cache_dir",
    "fetch_assets",
    "fetch_assets_main",
    "load_registry",
    "sha256_file",
    "status_summary",
    "verify",
]


@dataclass(frozen=True)
class AssetEntry:
    path: str  # POSIX relative path, e.g. "reg/mb101/lh.white.vtk"
    size: int
    sha256: str


@dataclass
class VerifyReport:
    ok: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    size_mismatch: list[str] = field(default_factory=list)
    hash_mismatch: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not (self.missing or self.size_mismatch or self.hash_mismatch)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def _load_raw() -> dict:
    text = resources.files("pybiscuit.data").joinpath(REGISTRY_RESOURCE).read_text("utf-8")
    return json.loads(text)


def load_registry() -> dict[str, AssetEntry]:
    """Return ``{relative_path: AssetEntry}`` from the committed registry."""
    raw = _load_raw()
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported asset registry schema {raw.get('schema_version')!r}")
    return {e["path"]: AssetEntry(e["path"], int(e["size"]), e["sha256"]) for e in raw["files"]}


def cache_dir() -> Path:
    """Default asset cache location (``$PYBISCUIT_ASSETS`` or ``~/.cache/pybiscuit/assets``)."""
    env = os.environ.get(CACHE_ENV)
    return Path(env) if env else Path.home() / ".cache" / "pybiscuit" / "assets"


def verify(root: str | os.PathLike, *, check_hashes: bool = True) -> VerifyReport:
    """Check directory ``root`` (laid out like ``/opt/biscuit``) against the registry."""
    root = Path(root)
    report = VerifyReport()
    for rel, entry in load_registry().items():
        p = root.joinpath(*rel.split("/"))
        if not p.is_file():
            report.missing.append(rel)
        elif p.stat().st_size != entry.size:
            report.size_mismatch.append(rel)
        elif check_hashes and sha256_file(p) != entry.sha256:
            report.hash_mismatch.append(rel)
        else:
            report.ok.append(rel)
    return report


def status_summary() -> str:
    """One-line registry/cache status for ``pybiscuit info``."""
    reg = load_registry()
    root = cache_dir()
    present = sum(
        1
        for rel, e in reg.items()
        if (p := root.joinpath(*rel.split("/"))).is_file() and p.stat().st_size == e.size
    )
    return f"assets: {len(reg)} files registered, {present} present in cache ({root})"


def fetch_assets(
    target_dir: str | os.PathLike | None = None,
    *,
    from_dir: str | os.PathLike | None = None,
    url: str | None = None,
    verify_only: bool = False,
    check_hashes: bool = True,
) -> VerifyReport:
    """Fetch and/or verify assets against the pinned asset registry."""
    import shutil
    import tarfile
    import zipfile

    target = Path(target_dir) if target_dir else cache_dir()
    target.mkdir(parents=True, exist_ok=True)

    if not verify_only:
        reg = load_registry()
        if from_dir:
            src = Path(from_dir)
            for rel in reg:
                s_path = src.joinpath(*rel.split("/"))
                d_path = target.joinpath(*rel.split("/"))
                if s_path.is_file():
                    d_path.parent.mkdir(parents=True, exist_ok=True)
                    if not d_path.is_file() or d_path.stat().st_size != s_path.stat().st_size:
                        shutil.copy2(s_path, d_path)
        elif url:
            import tempfile
            import urllib.request

            with tempfile.NamedTemporaryFile(suffix=".archive", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                urllib.request.urlretrieve(url, tmp_path)
                if tarfile.is_tarfile(tmp_path):
                    with tarfile.open(tmp_path, "r:*") as tf:
                        if hasattr(tarfile, "data_filter"):
                            tf.extractall(target, filter="data")
                        else:
                            tf.extractall(target)
                elif zipfile.is_zipfile(tmp_path):
                    with zipfile.ZipFile(tmp_path, "r") as zf:
                        zf.extractall(target)
                else:
                    raise ValueError(f"Unsupported archive format from {url}")
            finally:
                tmp_path.unlink(missing_ok=True)

    return verify(target, check_hashes=check_hashes)


def fetch_assets_main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for pybiscuit fetch-assets."""
    import argparse

    p = argparse.ArgumentParser(
        prog="pybiscuit fetch-assets",
        description="Verify or populate pyBiscuit model and template assets (B13.1).",
    )
    p.add_argument("-t", "--target", "--out", help="Target directory (default: cache_dir)")
    p.add_argument("--from-dir", help="Local directory containing extracted reference assets")
    p.add_argument("--url", help="URL of assets archive to download")
    p.add_argument("--verify-only", action="store_true", help="Only verify existing assets")
    p.add_argument("--quick", action="store_true", help="Skip SHA-256 computation (size-only)")
    args = p.parse_args(argv)

    target = Path(args.target) if args.target else cache_dir()
    rep = fetch_assets(
        target,
        from_dir=args.from_dir,
        url=args.url,
        verify_only=args.verify_only,
        check_hashes=not args.quick,
    )
    reg = load_registry()
    print(f"Target directory: {target}")
    print(f"Registered files: {len(reg)}")
    print(f"Verified OK:      {len(rep.ok)}")
    if rep.missing:
        print(f"Missing:          {len(rep.missing)}")
    if rep.size_mismatch:
        print(f"Size mismatch:    {len(rep.size_mismatch)}")
    if rep.hash_mismatch:
        print(f"Hash mismatch:    {len(rep.hash_mismatch)}")

    return 0 if rep.passed else 1
