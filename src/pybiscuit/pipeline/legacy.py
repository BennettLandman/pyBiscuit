"""Legacy BISCUIT output tree layout and population (`B10.4`, spec §13.1).

Mirrors the source-derived output tree of ``BISCUIT/GenericRecon/biscuit.m``
(lines 260-307, 343-347, 400-407, 457-468) and the ``run_biscuit`` container wrapper
(``PDF/biscuit.pdf`` merge, lines 199-202), including all ``gray``/``mid``/``CT``/``VertArea``
naming irregularities across ``surf/``, ``reg/``, ``roi/``, ``curve/``, ``quant/``, ``stat/``,
``pdf/``, and ``PDF/``.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

__all__ = [
    "HEMIS",
    "LEGACY_DIRS",
    "discover_legacy_files",
    "expected_legacy_files",
    "legacy_source_candidates",
    "normalize_protocols",
    "normalize_route",
    "populate_legacy_tree",
]

HEMIS: tuple[str, str] = ("lh", "rh")
LEGACY_DIRS: tuple[str, ...] = ("surf", "reg", "roi", "curve", "quant", "stat", "pdf", "PDF")


def normalize_route(route: str) -> str:
    """Normalize a route identifier to ``'macruise'``, ``'fs'``, or ``'civet'``."""
    r = route.strip().lower()
    if r in ("macruise", "mc", "pymacruise", "macruise-v3", "macruise_v3"):
        return "macruise"
    if r in ("fs", "freesurfer"):
        return "fs"
    if r in ("civet", "cv"):
        return "civet"
    raise ValueError(f"unknown route {route!r}; expected 'macruise', 'fs'/'freesurfer', or 'civet'")


def normalize_protocols(protocols: str | Iterable[str]) -> tuple[str, ...]:
    """Normalize a protocol list or comma-separated string into a tuple of protocol names."""
    if isinstance(protocols, str):
        parts = [p.strip() for p in protocols.split(",") if p.strip()]
    else:
        parts = [str(p).strip() for p in protocols if str(p).strip()]
    if not parts:
        raise ValueError("protocols must contain at least one protocol name")
    return tuple(parts)


def expected_legacy_files(
    protocols: str | Sequence[str] = ("braincolor", "dkt31"),
    reg_temp: str = "mb101",
    route: str = "macruise",
    include_merged_pdf: bool = True,
) -> set[str]:
    """Return the exact set of relative POSIX file paths in `$OUT` per spec §13.1.

    Parameters
    ----------
    protocols
        Parcellation protocols (default ``("braincolor", "dkt31")``).
    reg_temp
        Registration template name used in ``pdf/reg.<reg_temp>.pdf`` (default ``"mb101"``).
    route
        ``"macruise"`` (includes ``surf/hemi_map.txt``) or ``"fs"``/``"freesurfer"``/``"civet"``
        (omits ``surf/hemi_map.txt``).
    include_merged_pdf
        Whether to include ``PDF/biscuit.pdf`` (produced by the container wrapper ``run_biscuit``,
        not by ``biscuit.m`` itself).
    """
    norm_route = normalize_route(route)
    protos = normalize_protocols(protocols)
    reg_temp = reg_temp.strip()
    if not reg_temp:
        raise ValueError("reg_temp must be non-empty")

    files: set[str] = set()

    # surf/ (biscuit.m:280-286)
    for h in HEMIS:
        for s in ("white", "gray", "mid", "inflated", "sphere"):
            files.add(f"surf/{h}.{s}.vtk")
    if norm_route == "macruise":
        files.add("surf/hemi_map.txt")

    # reg/ (biscuit.m:269-277)
    for h in HEMIS:
        for s in ("white", "gray", "mid", "sphere"):
            files.add(f"reg/{h}.{s}.reg.vtk")
        files.add(f"reg/{h}.coeff.txt")

    # roi/ (biscuit.m:264-267)
    for h in HEMIS:
        files.add(f"roi/{h}.mask.txt")
        for p in protos:
            files.add(f"roi/{h}.parc.{p}.txt")

    # curve/ (biscuit.m:343-346)
    for h in HEMIS:
        files.add(f"curve/{h}.gray.scurve")
        files.add(f"curve/{h}.gray.scurve.bary")
        files.add(f"curve/{h}.white.gcurve")
        files.add(f"curve/{h}.white.gcurve.bary")

    # quant/ (biscuit.m:288-292, 403-405)
    for h in HEMIS:
        files.add(f"quant/{h}.thickness.txt")
        files.add(f"quant/{h}.mid.area.txt")
        files.add(f"quant/{h}.mid.H.txt")
        files.add(f"quant/{h}.mid.S.txt")
        files.add(f"quant/{h}.gray.SD.txt")
        files.add(f"quant/{h}.gray.LGI.txt")
        files.add(f"quant/{h}.mid.SCI.txt")

    # stat/ (biscuit.m:297-298, 460-468)
    for h in HEMIS:
        files.add(f"stat/{h}.mid.reg.area.txt")
        files.add(f"stat/{h}.mid.reg.H.txt")
        files.add(f"stat/{h}.mid.reg.S.txt")
        files.add(f"stat/{h}.mid.reg.SCI.txt")
        files.add(f"stat/{h}.gray.reg.LGI.txt")
        files.add(f"stat/{h}.gray.reg.SD.txt")
        files.add(f"stat/{h}.reg.thickness.txt")
    for p in protos:
        files.add(f"stat/stats.{p}.csv")

    # pdf/ (biscuit.m:304-307, 347, 407)
    files.add(f"pdf/reg.{reg_temp}.pdf")
    for p in protos:
        files.add(f"pdf/parc.{p}.pdf")
    files.add("pdf/curve.pdf")
    files.add("pdf/quant.pdf")

    # PDF/ (run_biscuit:199-202)
    if include_merged_pdf:
        files.add("PDF/biscuit.pdf")

    return files


def legacy_source_candidates(rel_path: str) -> tuple[str, ...]:
    """Return ordered candidate source names for a legacy relative path ``rel_path``.

    Encodes all BISCUIT scratch-to-output renames (``biscuit.m:260-307, 425-468``):
    - ``surf/<h>.gray.vtk`` <- ``<h>.gray.vtk``, ``<h>.pial.vtk``
    - ``reg/<h>.gray.reg.vtk`` <- ``<h>.gray.reg.vtk``, ``<h>.pial.reg.vtk``
    - ``quant/<h>.thickness.txt`` <- ``<h>.thickness.txt``, ``<h>.mid.CT.txt``
    - ``quant/<h>.mid.area.txt`` <- ``<h>.mid.area.txt``, ``<h>.mid.VertArea.txt``
    - ``quant/<h>.gray.SD.txt`` <- ``<h>.gray.SD.txt``, ``<h>.mid.SD.txt``
    - ``quant/<h>.gray.LGI.txt`` <- ``<h>.gray.LGI.txt``, ``<h>.mid.LGI.txt``
    - ``stat/<h>.mid.reg.area.txt`` <- ``<h>.mid.reg.area.txt``, ``<h>.mid.reg.VertArea.txt``
    - ``stat/<h>.gray.reg.LGI.txt`` <- ``<h>.gray.reg.LGI.txt``, ``<h>.mid.reg.LGI.txt``
    - ``stat/<h>.gray.reg.SD.txt`` <- ``<h>.gray.reg.SD.txt``, ``<h>.mid.reg.SD.txt``
    - ``stat/<h>.reg.thickness.txt`` <- ``<h>.reg.thickness.txt``, ``<h>.mid.reg.CT.txt``
    """
    base = Path(rel_path).name
    cands: list[str] = [rel_path, base]
    for h in HEMIS:
        if rel_path == f"surf/{h}.gray.vtk":
            cands.extend([f"surf/{h}.pial.vtk", f"{h}.pial.vtk"])
        elif rel_path == f"reg/{h}.gray.reg.vtk":
            cands.extend([f"reg/{h}.pial.reg.vtk", f"{h}.pial.reg.vtk"])
        elif rel_path == f"quant/{h}.thickness.txt":
            cands.extend([f"quant/{h}.mid.CT.txt", f"{h}.mid.CT.txt"])
        elif rel_path == f"quant/{h}.mid.area.txt":
            cands.extend([f"quant/{h}.mid.VertArea.txt", f"{h}.mid.VertArea.txt"])
        elif rel_path == f"quant/{h}.gray.SD.txt":
            cands.extend([f"quant/{h}.mid.SD.txt", f"{h}.mid.SD.txt"])
        elif rel_path == f"quant/{h}.gray.LGI.txt":
            cands.extend([f"quant/{h}.mid.LGI.txt", f"{h}.mid.LGI.txt"])
        elif rel_path == f"stat/{h}.mid.reg.area.txt":
            cands.extend([f"stat/{h}.mid.reg.VertArea.txt", f"{h}.mid.reg.VertArea.txt"])
        elif rel_path == f"stat/{h}.gray.reg.LGI.txt":
            cands.extend([f"stat/{h}.mid.reg.LGI.txt", f"{h}.mid.reg.LGI.txt"])
        elif rel_path == f"stat/{h}.gray.reg.SD.txt":
            cands.extend([f"stat/{h}.mid.reg.SD.txt", f"{h}.mid.reg.SD.txt"])
        elif rel_path == f"stat/{h}.reg.thickness.txt":
            cands.extend([f"stat/{h}.mid.reg.CT.txt", f"{h}.mid.reg.CT.txt"])
    # deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return tuple(out)


def discover_legacy_files(root: str | Path) -> set[str]:
    """Discover BISCUIT legacy output files under ``root``.

    Handles both standalone BISCUIT output directories and in-place FreeSurfer subject
    directories (such as ``open-examples/0050952``) where ``surf/`` also holds native FreeSurfer
    files: under ``surf/`` only ``*.vtk`` and ``hemi_map.txt`` are collected, while under
    ``reg/``, ``roi/``, ``curve/``, ``quant/``, ``stat/``, ``pdf/``, and ``PDF/`` all non-hidden
    files are collected. Also handles case-insensitive filesystems (macOS APFS) where ``pdf/``
    and ``PDF/`` share a directory entry by mapping ``biscuit.pdf`` to ``PDF/biscuit.pdf``.
    """
    root = Path(root)
    if not root.is_dir():
        return set()

    actual_entries = set(os.listdir(root))
    found: set[str] = set()

    if "surf" in actual_entries and (root / "surf").is_dir():
        for f in (root / "surf").iterdir():
            if f.is_file() and not f.name.startswith("."):
                if f.suffix == ".vtk" or f.name == "hemi_map.txt":
                    found.add(f"surf/{f.name}")

    for d in ("reg", "roi", "curve", "quant", "stat"):
        if d in actual_entries and (root / d).is_dir():
            for f in (root / d).rglob("*"):
                if f.is_file():
                    rel = f.relative_to(root)
                    if not any(part.startswith(".") for part in rel.parts):
                        found.add(rel.as_posix())

    for pdf_dir in ("pdf", "PDF"):
        if pdf_dir in actual_entries and (root / pdf_dir).is_dir():
            for f in (root / pdf_dir).rglob("*"):
                if f.is_file():
                    rel = f.relative_to(root / pdf_dir)
                    if any(part.startswith(".") for part in rel.parts):
                        continue
                    if rel.as_posix() == "biscuit.pdf":
                        found.add("PDF/biscuit.pdf")
                    else:
                        found.add(f"pdf/{rel.as_posix()}")

    return found


def _resolve_source(
    rel_path: str,
    sources: Mapping[str, str | Path] | Sequence[str | Path] | str | Path,
) -> Path | None:
    cands = legacy_source_candidates(rel_path)
    if isinstance(sources, Mapping):
        for c in cands:
            if c in sources:
                p = Path(sources[c])
                if p.is_file():
                    return p
        return None
    search_dirs = (
        [Path(sources)] if isinstance(sources, (str, Path)) else [Path(d) for d in sources]
    )
    for d in search_dirs:
        for c in cands:
            p = d / c
            if p.is_file():
                return p
            # also check pdf/ subdirectory inside scratch dirs (e.g. tmp1/pdf/reg.mb101.pdf)
            if rel_path.startswith("pdf/"):
                p_pdf = d / "pdf" / Path(c).name
                if p_pdf.is_file():
                    return p_pdf
    return None


def _place_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_file() and not dst.is_symlink() and os.path.samefile(src, dst):
            return
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"unsupported mode {mode!r}; expected 'copy', 'hardlink', or 'symlink'")


def populate_legacy_tree(
    legacy_dir: str | Path,
    sources: Mapping[str, str | Path] | Sequence[str | Path] | str | Path,
    *,
    protocols: str | Sequence[str] = ("braincolor", "dkt31"),
    reg_temp: str = "mb101",
    route: str = "macruise",
    include_merged_pdf: bool = True,
    mode: str = "copy",
    allow_partial: bool = False,
) -> dict[str, Path]:
    """Populate ``legacy_dir`` with the BISCUIT legacy output view (spec §13.1).

    ``sources`` may be:
    - a mapping ``{key: path}`` where ``key`` is either the target legacy relative path
      (e.g. ``"stat/lh.reg.thickness.txt"``), the target basename, or a BISCUIT scratch
      intermediate name (e.g. ``"lh.mid.reg.CT.txt"``, ``"lh.pial.vtk"``,
      ``"lh.mid.VertArea.txt"``),
    - a single directory path (searched by relative path, basename, and scratch aliases), or
    - a sequence of directory paths (e.g. ``[tmp1, tmp2, tmp3, tmp4]``).

    Returns ``{rel_path: destination_path}`` for all populated files.
    """
    legacy_dir = Path(legacy_dir)
    expected = sorted(
        expected_legacy_files(
            protocols=protocols,
            reg_temp=reg_temp,
            route=route,
            include_merged_pdf=include_merged_pdf,
        ),
        key=lambda rel: (rel.startswith("PDF/"), rel),
    )
    populated: dict[str, Path] = {}
    missing: list[str] = []

    for rel in expected:
        src = _resolve_source(rel, sources)
        if src is None:
            missing.append(rel)
            continue
        dst = legacy_dir / rel
        _place_file(src, dst, mode)
        populated[rel] = dst

    if missing and not allow_partial:
        raise FileNotFoundError(
            f"missing {len(missing)} expected legacy file(s) when populating {legacy_dir}: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    return populated
