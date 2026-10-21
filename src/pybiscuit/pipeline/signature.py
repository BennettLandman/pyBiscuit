"""Content-addressed stage signatures.

A stage signature is the SHA-256 of canonical JSON over: the stage name and
version, the pybiscuit version, the hashes of its adapter inputs, the signatures
of the upstream stages it consumes (standing in for their outputs), the values of
its declared config keys, and the hashes of its declared assets. Anything that can
change a stage's result therefore changes its signature; nothing else does
(timestamps, paths of the output directory, thread count).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .. import __version__, assets
from ..config import Config
from .model import InputFindings, PipelineError, Stage

__all__ = ["ASSET_PREFIXES", "asset_hashes", "canonical_json", "components", "digest"]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


# asset key -> function(config) -> registry path prefixes the stage depends on
ASSET_PREFIXES = {
    "reg_template": lambda c: [f"reg/{c['reg_temp']}/"],
    "parc_models": lambda c: [f"parc/{p}/" for p in c["parc_protocol"]],
    "icospheres": lambda c: ["sphere/"],
}


def asset_hashes(keys: tuple[str, ...], config: Config) -> dict[str, str]:
    """Digest of the registry entries behind each asset key (registry, not disk, so
    plans and dry-runs work without the assets installed)."""
    reg = assets.load_registry()
    out = {}
    for key in keys:
        prefixes = ASSET_PREFIXES[key](config)
        entries = {p: reg[p].sha256 for pre in prefixes for p in reg if p.startswith(pre)}
        for pre in prefixes:
            if not any(p.startswith(pre) for p in entries):
                raise PipelineError(f"asset {key}: nothing in the registry under {pre!r}")
        out[key] = digest(entries)
    return out


def components(
    stage: Stage,
    findings: InputFindings,
    config: Config,
    upstream_signatures: dict[str, str],
) -> dict[str, Any]:
    """The exact dict a signature is computed over (also stored in ``run.json`` so a
    refused resume can say what changed)."""
    inputs: dict[str, str] = {}
    for name in stage.inputs:
        if name == "*":
            inputs.update({f"input:{n}": h for n, h in findings.hashes().items()})
        elif ":" in name:
            up = name.split(":", 1)[0]
            if up in upstream_signatures:  # absent on routes without that stage
                inputs[name] = upstream_signatures[up]
        elif name in findings.products:
            inputs[f"input:{name}"] = findings.products[name].sha256
        else:
            inputs[f"input:{name}"] = "<absent>"
    return {
        "stage": stage.name,
        "stage_version": stage.version,
        "pybiscuit": __version__,
        "inputs": inputs,
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k in stage.params
                   for v in [config[k]]},
        "assets": asset_hashes(stage.assets, config),
        "outputs": list(stage.outputs),
    }
