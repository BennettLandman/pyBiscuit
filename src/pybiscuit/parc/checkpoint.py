"""Safe conversion of the legacy parcellation checkpoints to safetensors (plan row B7.2).

The shipped ``parc/<protocol>/?h*.pth`` files are third-party **legacy (non-zip) torch pickles**
(protocol 2, magic ``\\x80\\x02\\x8a\\x0a``).  Rules (brief 06):

* never ``torch.load(..., weights_only=False)``; :func:`load_legacy_state_dict` first audits every
  opcode/global with ``pickletools`` (no execution) and then calls
  ``torch.load(weights_only=True)``, whose unpickler is itself restricted to tensor rebuilds;
* the checkpoint is converted once (:func:`convert_checkpoint`), the SHA-256 of source and result
  are recorded, and runtime only ever reads the safetensors (:func:`load_safetensors`).

Key canonicalisation.  The legacy ``state_dict`` lists every ``ResBlock`` layer twice (attribute
name and ``seq1.K``/``seq2.K`` position; the two entries share one storage) and carries the
``NS``/``EW`` operator buffers.  :func:`canonicalize_state_dict` maps each layer to one canonical
name (``conv1, bn1, conv2, bn2, conv3, bn3, conv_, bn_``), asserts aliases are bit-identical and
drops ``NS``/``EW`` after proving that they equal the regenerated operators (they are derived data).
"""

from __future__ import annotations

import collections
import hashlib
import json
import pickletools
import re
from collections.abc import Callable, Mapping
from importlib import resources
from pathlib import Path

import numpy as np

__all__ = [
    "ALLOWED_GLOBALS",
    "LEGACY_MAGIC",
    "UnsafeCheckpointError",
    "audit_legacy_pickle",
    "canonicalize_state_dict",
    "convert_checkpoint",
    "load_legacy_state_dict",
    "load_safetensors",
    "converted_registry",
]

LEGACY_MAGIC = b"\x80\x02\x8a\x0a"
FORMAT = "pybiscuit-parc-unet/1"

#: The only globals present in the 12 shipped checkpoints (``pickletools`` audit, 2026-10-20).
ALLOWED_GLOBALS = frozenset(
    {
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch", "FloatStorage"),
        ("collections", "OrderedDict"),
    }
)
#: Opcodes that can build arbitrary objects; none occur in the shipped files.
_FORBIDDEN_OPS = frozenset(
    {"INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "BUILD", "STACK_GLOBAL", "EXT1", "EXT2", "EXT4"}
)


class UnsafeCheckpointError(RuntimeError):
    """The file is not a legacy torch checkpoint made only of the audited constructs."""


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- audit


def audit_legacy_pickle(path: str | Path) -> dict:
    """Statically audit a legacy ``.pth`` (nothing is executed or unpickled).

    A legacy file is five consecutive pickles (magic number, protocol version, system info,
    ``{'state_dict': ...}``, storage keys) followed by raw storage bytes.  Returns the global
    references found; raises :class:`UnsafeCheckpointError` for anything outside
    :data:`ALLOWED_GLOBALS` or any object-constructing opcode.
    """
    globs: collections.Counter = collections.Counter()
    with open(path, "rb") as fh:
        if fh.read(4) != LEGACY_MAGIC:
            raise UnsafeCheckpointError(f"{path}: not a legacy (non-zip) torch pickle")
        fh.seek(0)
        for part in range(5):
            try:
                for op, arg, _ in pickletools.genops(fh):
                    if op.name in _FORBIDDEN_OPS:
                        raise UnsafeCheckpointError(f"{path}: forbidden opcode {op.name}")
                    if op.name == "GLOBAL":
                        mod, name = str(arg).split(" ", 1)
                        if (mod, name) not in ALLOWED_GLOBALS:
                            raise UnsafeCheckpointError(f"{path}: global {mod}.{name} not allowed")
                        globs[f"{mod}.{name}"] += 1
            except UnsafeCheckpointError:
                raise
            except Exception as exc:  # malformed stream
                raise UnsafeCheckpointError(f"{path}: pickle {part} unreadable: {exc}") from exc
    return {"globals": dict(globs)}


def load_legacy_state_dict(path: str | Path) -> dict:
    """Audit, then load the ``state_dict`` of a legacy checkpoint with ``weights_only=True``."""
    import torch

    audit_legacy_pickle(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(obj, Mapping) or "state_dict" not in obj:
        raise UnsafeCheckpointError(f"{path}: no 'state_dict' entry")
    return dict(obj["state_dict"])


# --------------------------------------------------------------------------- canonical keys

# position of each layer inside ResBlock.seq1 / seq2 (coarsening blocks have the extra DownSamp)
_SEQ_NAMES = {
    True: {
        "seq1": {0: "conv1", 2: "bn1", 4: "conv2", 5: "bn2", 7: "conv3", 8: "bn3"},
        "seq2": {0: "conv_", 2: "bn_"},
    },
    False: {
        "seq1": {0: "conv1", 1: "bn1", 3: "conv2", 4: "bn2", 6: "conv3", 7: "bn3"},
        "seq2": {0: "conv_", 1: "bn_"},
    },
}
_DERIVED = ("NS", "EW")  # operator buffers: regenerated, not weights
_KEY = re.compile(r"^(?P<pre>(?:down|up)\.\d+\.conv)\.(?P<rest>.+)$")


def _canonical_name(key: str) -> str:
    m = _KEY.match(key)
    if not m:
        return key
    coarsen = key.startswith("down.")
    head, _, tail = m["rest"].partition(".")
    if head in ("seq1", "seq2"):
        idx, _, leaf = tail.partition(".")
        try:
            return f"{m['pre']}.{_SEQ_NAMES[coarsen][head][int(idx)]}.{leaf}"
        except (KeyError, ValueError):
            raise UnsafeCheckpointError(f"unexpected layer position in key {key!r}") from None
    return key


def canonicalize_state_dict(
    state_dict: Mapping,
    ops_for_level: Callable[[int], dict] | None = None,
    *,
    ns_ew_tol: float = 1e-6,
) -> tuple[dict, dict]:
    """Legacy (or current-``ops.py``) ``state_dict`` -> ``(canonical float32 tensors, report)``.

    * a leading ``module.`` (``cpuModel`` wrapper) is stripped;
    * aliased layers are merged; aliases must be bit-identical;
    * ``NS``/``EW`` are removed after checking them against ``ops_for_level(level)`` within
      ``ns_ew_tol`` (default: the regenerated operators; ``level`` follows from the face count);
    * ``num_batches_tracked`` (if present) is dropped: it only matters for training.
    """
    import torch

    if ops_for_level is None:
        from ..sphere.operators import build_operators as ops_for_level
    out: dict = {}
    n_alias = 0
    max_ns_diff = 0.0
    for key, t in state_dict.items():
        k = key[len("module.") :] if key.startswith("module.") else key
        if k.endswith("num_batches_tracked"):
            continue
        if k.rsplit(".", 1)[-1] in _DERIVED:
            nf = int(t.shape[0])
            level = round(np.log(nf / 20.0) / np.log(4.0))
            if 20 * 4**level != nf:
                raise UnsafeCheckpointError(f"{key}: {nf} faces is not an icosphere")
            ref = np.asarray(ops_for_level(level)[k.rsplit(".", 1)[-1]], dtype=np.float64)
            d = float(np.abs(t.detach().double().numpy() - ref).max())
            if d > ns_ew_tol:
                raise UnsafeCheckpointError(f"{key}: differs from regenerated operator ({d:.3g})")
            max_ns_diff = max(max_ns_diff, d)
            continue
        if not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or t.layout != torch.strided:
            raise UnsafeCheckpointError(f"{key}: expected a dense float32 tensor")
        name = _canonical_name(k)
        t = t.detach().contiguous()
        if name in out:
            if not torch.equal(out[name], t):
                raise UnsafeCheckpointError(f"{key}: alias of {name!r} with different values")
            n_alias += 1
        else:
            out[name] = t.clone()
    return out, {"aliases_merged": n_alias, "ns_ew_max_abs_diff": max_ns_diff}


# --------------------------------------------------------------------------- convert / load


def _registry_key(src: Path) -> str:
    return f"parc/{src.parent.name}/{src.name}"


def converted_registry() -> dict:
    """The committed record of conversions: ``{relative safetensors path: entry}``."""
    text = resources.files("pybiscuit.data").joinpath("parc_converted_registry.json").read_text()
    raw = json.loads(text)
    return {e["path"]: e for e in raw["files"]}


def convert_checkpoint(
    src: str | Path,
    dst: str | Path,
    *,
    expected_sha256: str | None = None,
    ops_for_level: Callable[[int], dict] | None = None,
) -> dict:
    """Convert one legacy ``.pth`` into a safetensors file; returns the registry entry.

    ``expected_sha256`` (normally the asset-registry value for ``src``) is checked *before* the
    file is opened for unpickling.  The output is deterministic (no torch version in the file).
    """
    import torch
    from safetensors.torch import save_file

    from ..assets import load_registry
    from .unet import infer_architecture

    src, dst = Path(src), Path(dst)
    digest = sha256_file(src)
    if expected_sha256 is None:
        reg = load_registry().get(_registry_key(src))
        expected_sha256 = reg.sha256 if reg else None
    if expected_sha256 is not None and digest != expected_sha256:
        raise UnsafeCheckpointError(f"{src}: sha256 {digest} != expected {expected_sha256}")
    audit = audit_legacy_pickle(src)
    sd = load_legacy_state_dict(src)
    canon, report = canonicalize_state_dict(sd, ops_for_level)
    arch = infer_architecture(canon)
    meta = {
        "format": FORMAT,
        "source": _registry_key(src),
        "source_sha256": digest,
        "arch": arch,
        "dropped": "NS,EW (operator buffers, equal to regenerated operators), num_batches_tracked",
        "terms": "derived from third-party weights; redistribution permission unresolved "
        "(open-questions Q6/Q7); do not publish",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    # one metadata entry: safetensors writes the metadata map in hash order, so several entries
    # would make the file bytes (and its SHA-256) vary between runs
    save_file(
        {k: canon[k] for k in sorted(canon)},
        str(tmp),
        metadata={"pybiscuit": json.dumps(meta, sort_keys=True)},
    )
    tmp.replace(dst)
    # read back and compare bit for bit with what was converted
    back = load_safetensors(dst)[0]
    for k, v in canon.items():
        if not torch.equal(back[k], v):
            raise RuntimeError(f"{dst}: round trip changed {k}")
    return {
        "path": f"{_registry_key(src)[: -len('.pth')]}.safetensors",
        "size": dst.stat().st_size,
        "sha256": sha256_file(dst),
        "source": _registry_key(src),
        "source_sha256": digest,
        "tensors": len(canon),
        "arch": arch,
        "globals_audited": audit["globals"],
        **report,
        "torch": torch.__version__,
    }


def load_safetensors(path: str | Path, *, expected_sha256: str | None = None) -> tuple[dict, dict]:
    """Read a converted checkpoint: ``(tensors, metadata)``.  Optionally verify its SHA-256."""
    from safetensors import safe_open

    path = Path(path)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"{path}: sha256 mismatch")
    tensors: dict = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        raw = (f.metadata() or {}).get("pybiscuit")
        meta = json.loads(raw) if raw else {}
        for k in f.keys():  # noqa: SIM118 - safetensors handle, not a dict
            tensors[k] = f.get_tensor(k)
    if meta.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    return tensors, meta
