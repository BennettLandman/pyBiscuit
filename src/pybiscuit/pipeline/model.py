"""Data model of the stage graph: products, input findings, stages, errors."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "InputError",
    "InputFindings",
    "PipelineError",
    "Product",
    "ResumeError",
    "Stage",
    "StageBlocked",
    "StageContext",
    "file_record",
    "sha256_file",
]


class PipelineError(Exception):
    """Base class of pipeline failures that carry a user-facing message."""


class StageBlocked(PipelineError):
    """A stage cannot run yet; the message cites the plan row(s) and the reason."""


class ResumeError(PipelineError):
    """``--resume`` refused: recorded state does not match the current request."""


class InputError(PipelineError):
    """The input directory is unusable (incomplete, inconsistent, wrong frame)."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def file_record(path: str | Path, **extra: Any) -> dict[str, Any]:
    """``{path, sha256, bytes, ...}`` record, the shape PyMaCRUISE uses in ``final_outputs``."""
    p = Path(path)
    return {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size, **extra}


@dataclass(frozen=True)
class Product:
    """One located input file."""

    name: str  # logical name, e.g. "inner_surface"
    path: str
    sha256: str
    bytes: int
    kind: str  # surface | volume | transform | label | table | other
    frame: str | None = None  # coordinate frame of the data ("RAS", "tkRAS", "MNI", ...)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "path": self.path,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "kind": self.kind,
        }
        if self.frame:
            d["frame"] = self.frame
        if self.meta:
            d["meta"] = self.meta
        return d


@dataclass
class InputFindings:
    """What an input adapter found. ``ok`` is False when a required product is missing
    or a consistency check failed."""

    adapter: str
    root: str
    surf_recon: str  # route this input implies: "macruise" | "freesurfer"
    products: dict[str, Product] = field(default_factory=dict)
    required: tuple[str, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)  # upstream version/signature/status
    frames: dict[str, str] = field(default_factory=dict)  # product name -> frame
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        return [n for n in self.required if n not in self.products]

    @property
    def ok(self) -> bool:
        return not self.errors and not self.missing

    def problems(self) -> list[str]:
        return [*self.errors, *(f"missing required product: {n}" for n in self.missing)]

    def require_ok(self) -> None:
        if not self.ok:
            raise InputError(
                f"{self.adapter} input {self.root} is not usable:\n  - "
                + "\n  - ".join(self.problems())
            )

    def hashes(self) -> dict[str, str]:
        return {n: p.sha256 for n, p in sorted(self.products.items())}

    def to_json(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "root": self.root,
            "surf_recon": self.surf_recon,
            "source": self.source,
            "frames": self.frames,
            "products": {n: p.to_json() for n, p in sorted(self.products.items())},
            "warnings": self.warnings,
        }


@dataclass
class StageContext:
    """Passed to a stage body."""

    stage: Stage
    attempt_dir: Path
    inputs: dict[str, Path]  # declared input name -> file path
    params: dict[str, Any]
    config: Any  # pybiscuit.config.Config
    findings: InputFindings


@dataclass(frozen=True)
class Stage:
    """A node of the stage graph.

    ``inputs`` are adapter product names (``"t1"``), ``"*"`` for all of them, or
    ``"<stage>:<output>"`` for an upstream stage output. ``params`` are config keys
    that affect results (threads deliberately excluded). ``assets`` are keys of
    :data:`pybiscuit.pipeline.signature.ASSET_PREFIXES`. A stage with ``blocked`` set
    has no body yet; running it raises :class:`StageBlocked` with that text.
    """

    name: str
    plan_rows: tuple[str, ...]
    inputs: tuple[str, ...] = ()
    params: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()
    routes: tuple[str, ...] = ("macruise", "freesurfer")
    version: str = "1"  # bump when the stage body changes behaviour
    body: Callable[[StageContext], None] | None = None
    blocked: str | None = None

    def upstream(self) -> list[str]:
        return sorted({i.split(":", 1)[0] for i in self.inputs if ":" in i})

    def product_inputs(self) -> list[str]:
        return [i for i in self.inputs if ":" not in i]
