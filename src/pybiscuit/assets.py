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
