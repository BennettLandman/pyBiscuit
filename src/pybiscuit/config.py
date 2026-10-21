"""BISCUIT configuration: ``biscuit.conf`` and TOML readers, defaults, CLI overrides.

The format and key set follow the original ``/opt/biscuit/biscuit.conf`` of
``biscuit_FC_v2.2.sif`` (``key = value`` lines, ``%`` comment lines). The
``run_biscuit`` wrapper overrides individual keys from named flags and drops the
``fs_``/``mc_``/``cv_`` blocks that do not belong to the selected
``surf_recon``; both behaviours are reproduced here (:data:`CLI_FLAGS`,
:meth:`Config.active_items`).

Precedence, lowest to highest: built-in defaults < ``biscuit.conf`` < TOML <
CLI overrides (see :func:`load_config`).
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib

__all__ = [
    "CLI_FLAGS",
    "DEFAULTS",
    "SURF_RECON_CHOICES",
    "Config",
    "ConfigError",
    "load_config",
    "read_conf",
    "read_toml",
]

SURF_RECON_CHOICES = ("freesurfer", "macruise", "civet")


class ConfigError(ValueError):
    """Invalid configuration file, key, or value."""


# Key -> (default, type). Order is the order of the shipped file, which the
# writer reproduces. Defaults are those of the FC image's biscuit.conf.
_SCHEMA: dict[str, tuple[Any, str]] = {
    "project": ("", "str"),
    "subject": ("", "str"),
    "session": ("", "str"),
    "omp_num_threads": (1, "int"),
    "surf_recon": ("freesurfer", "recon"),
    "fs_surf_model_path": ("surf", "str"),
    "fs_label_model_path": ("label", "str"),
    "mc_surf": ("target_image_GMimg_innerSurf.asc", "str"),
    "mc_surf_mni": ("target_image_GMimg_innerSurf_mni.asc", "str"),
    "mc_rawT1": ("T1.nii.gz", "str"),
    "mc_rawSeg": ("SEG_refine.nii.gz", "str"),
    "cv_prefix_id": ("", "str"),
    "cv_surf_path": ("NonCorrespond", "str"),
    "cv_surf_postfix": ("_native_itkspace", "str"),
    "cv_surf_sphere": (327680, "int"),
    "cv_ct_path": ("Thickness", "str"),
    "cv_ct_method": ("tlink_20mm", "str"),
    "surf_smooth": (0, "int"),
    "reg_temp": ("mb101", "str"),
    "parc_protocol": (("braincolor", "dkt31"), "list"),
    "trace_r": (4, "num"),
    "trace_delta": (2.5, "num"),
    "curv_smoothness": (2, "int"),
    "lgi_kernel_size": (316, "int"),
    "lgi_anisotropy": (0.2, "num"),
    "lgi_ref_area": (77100, "num"),
    "sci_kernel_size": (3, "int"),
    "sci_bins": (5, "int"),
}

DEFAULTS: dict[str, Any] = {k: v[0] for k, v in _SCHEMA.items()}

# Section layout of the shipped file: (comment, keys).
_SECTIONS: list[tuple[str, tuple[str, ...]]] = [
    ("processing information", ("project", "subject", "session")),
    ("OpenMP threads", ("omp_num_threads",)),
    ("surface reconstruction type", ("surf_recon",)),
    ("freesurfer", ("fs_surf_model_path", "fs_label_model_path")),
    ("macruise", ("mc_surf", "mc_surf_mni", "mc_rawT1", "mc_rawSeg")),
    (
        "civet",
        (
            "cv_prefix_id",
            "cv_surf_path",
            "cv_surf_postfix",
            "cv_surf_sphere",
            "cv_ct_path",
            "cv_ct_method",
        ),
    ),
    ("surface smoothing (# of iterations)", ("surf_smooth",)),
    ("registration template", ("reg_temp",)),
    ("parcellation protocol", ("parc_protocol",)),
    ("parameters for TRACE", ("trace_r", "trace_delta")),
    (
        "parameters for LGI and SCI",
        (
            "curv_smoothness",
            "lgi_kernel_size",
            "lgi_anisotropy",
            "lgi_ref_area",
            "sci_kernel_size",
            "sci_bins",
        ),
    ),
]

# run_biscuit flag (without leading dashes) -> conf key.
CLI_FLAGS: dict[str, str] = {
    "proj": "project",
    "subj": "subject",
    "sess": "session",
    "thread": "omp_num_threads",
    "surf_recon": "surf_recon",
    "reg": "reg_temp",
    "parc": "parc_protocol",
    "trace_r": "trace_r",
    "trace_delta": "trace_delta",
    "curv_smooth": "curv_smoothness",
    "lgi_kernel": "lgi_kernel_size",
    "lgi_aniso": "lgi_anisotropy",
    "lgi_ref": "lgi_ref_area",
    "sci_kernel": "sci_kernel_size",
    "sci_bins": "sci_bins",
}

# Block prefix that run_biscuit strips when surf_recon selects another route.
_RECON_PREFIX = {"freesurfer": "fs_", "macruise": "mc_", "civet": "cv_"}


def _parse_num(key: str, text: str) -> int | float:
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        raise ConfigError(f"{key}: expected a number, got {text!r}") from None


def _coerce(key: str, value: Any) -> Any:
    """Convert a raw (string or TOML-typed) value to the key's canonical type."""
    if key not in _SCHEMA:
        raise ConfigError(f"unknown configuration key {key!r}")
    kind = _SCHEMA[key][1]
    if kind == "list":
        if isinstance(value, str):
            # run_biscuit strips all blanks before use ("braincolor, dkt31" works)
            value = [t for t in value.replace(" ", "").split(",")]
        items = tuple(str(t).strip() for t in value)
        if not items or any(not t for t in items):
            raise ConfigError(f"{key}: expected a non-empty comma separated list, got {value!r}")
        return items
    if isinstance(value, bool):
        raise ConfigError(f"{key}: booleans are not valid values")
    if kind == "str":
        return str(value).strip()
    if kind == "recon":
        v = str(value).strip().lower()  # run_biscuit lower-cases --surf_recon
        if v not in SURF_RECON_CHOICES:
            raise ConfigError(f"{key}: must be one of {SURF_RECON_CHOICES}, got {value!r}")
        return v
    if isinstance(value, str):
        value = _parse_num(key, value.strip())
    if kind == "int":
        if isinstance(value, float):
            if not value.is_integer():
                raise ConfigError(f"{key}: expected an integer, got {value!r}")
            value = int(value)
        return int(value)
    return value  # "num": int or float as given


def read_conf(path: str | Path) -> dict[str, str]:
    """Parse an original ``biscuit.conf`` into ``{key: raw string}`` (file order).

    Lines starting with ``%`` and blank lines are ignored; ``key = value`` is split
    at the first ``=`` and both sides are stripped (the wrapper strips blanks too).
    Unknown keys are rejected so typos do not pass silently.
    """
    out: dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        key, sep, value = s.partition("=")
        if not sep:
            raise ConfigError(f"{path}:{n}: expected 'key = value', got {line!r}")
        key = key.strip()
        if key not in _SCHEMA:
            raise ConfigError(f"{path}:{n}: unknown configuration key {key!r}")
        out[key] = value.strip()
    return out


def read_toml(path: str | Path) -> dict[str, Any]:
    """Parse a TOML config. Keys are the ``biscuit.conf`` names, at top level or in
    a ``[biscuit]`` table; values keep their TOML types."""
    with open(path, "rb") as fh:
        try:
            data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    table = data.get("biscuit", data)
    if not isinstance(table, dict):
        raise ConfigError(f"{path}: [biscuit] must be a table")
    stray = [k for k, v in data.items() if isinstance(v, dict) and k != "biscuit"]
    if stray:
        raise ConfigError(f"{path}: unknown table(s) {stray}; use top-level keys or [biscuit]")
    return dict(table)


@dataclass(frozen=True)
class Config:
    """Resolved configuration: a validated ``{key: value}`` map with provenance.

    ``sources`` lists where values came from, lowest to highest precedence, for
    ``run.json`` and ``--dry-run``.
    """

    values: Mapping[str, Any]
    sources: tuple[str, ...] = ("defaults",)

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def replace(self, **updates: Any) -> Config:
        merged = dict(self.values)
        for k, v in updates.items():
            merged[k] = _coerce(k, v)
        return Config(merged, self.sources)

    def active_items(self) -> dict[str, Any]:
        """Items ``run_biscuit`` passes on: the other routes' prefixed blocks are dropped."""
        recon = self.values["surf_recon"]
        drop = tuple(p for r, p in _RECON_PREFIX.items() if r != recon)
        return {k: v for k, v in self.values.items() if not k.startswith(drop)}

    def to_dict(self, *, active_only: bool = False) -> dict[str, Any]:
        """JSON-serialisable dict (lists for tuples)."""
        items = self.active_items() if active_only else dict(self.values)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in items.items()}

    def to_conf(self, *, active_only: bool = False) -> str:
        """Render in the original ``biscuit.conf`` layout (what ``run_biscuit`` writes)."""
        items = self.active_items() if active_only else dict(self.values)
        blocks = []
        for title, keys in _SECTIONS:
            lines = [f"{k} = {_fmt(items[k])}".rstrip() for k in keys if k in items]
            if lines:
                blocks.append("\n".join([f"% {title}", *lines]))
        return "\n\n".join(blocks) + "\n"


def _fmt(value: Any) -> str:
    if isinstance(value, tuple):
        return ",".join(value)
    return str(value)


def load_config(
    conf: str | Path | None = None,
    toml: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Merge defaults, an optional ``biscuit.conf``, an optional TOML file, and overrides.

    ``overrides`` keys may be conf keys (``reg_temp``) or ``run_biscuit`` flag names
    (``reg``); ``None`` values are skipped, so argparse namespaces can be passed
    directly.
    """
    values = dict(DEFAULTS)
    sources = ["defaults"]
    if conf is not None:
        for k, v in read_conf(conf).items():
            values[k] = _coerce(k, v)
        sources.append(f"conf:{conf}")
    if toml is not None:
        for k, v in read_toml(toml).items():
            values[k] = _coerce(k, v)
        sources.append(f"toml:{toml}")
    applied = []
    for flag, v in (overrides or {}).items():
        if v is None:
            continue
        key = CLI_FLAGS.get(flag, flag)
        values[key] = _coerce(key, v)
        applied.append(key)
    if applied:
        sources.append("overrides:" + ",".join(applied))
    return Config(values, tuple(sources))
