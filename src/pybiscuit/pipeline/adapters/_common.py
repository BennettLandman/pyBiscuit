"""Helpers shared by the input adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..model import InputFindings, Product, sha256_file


def make_product(
    name: str,
    path: Path,
    kind: str,
    frame: str | None = None,
    *,
    findings: InputFindings,
    recorded_sha256: str | None = None,
    **meta,
) -> Product | None:
    """Hash ``path`` and add it to ``findings``; check it against a recorded hash if given."""
    digest = sha256_file(path)
    if recorded_sha256 is not None and digest != recorded_sha256:
        findings.errors.append(
            f"{name}: {path} no longer matches its recorded SHA-256 "
            f"({recorded_sha256[:12]} recorded, {digest[:12]} on disk)"
        )
        return None
    prod = Product(name, str(path), digest, path.stat().st_size, kind, frame, dict(meta))
    findings.products[name] = prod
    if frame:
        findings.frames[name] = frame
    return prod


def volume_info(path: Path) -> dict:
    """Shape, axis codes and affine of a NIfTI/MGH volume (header only)."""
    import nibabel as nib

    img = nib.load(str(path))
    return {
        "shape": [int(s) for s in img.shape],
        "axis_codes": list(nib.aff2axcodes(img.affine)),
        "affine": np.asarray(img.affine, dtype=float),
    }


def check_same_grid(
    findings: InputFindings, names: list[str], *, atol: float = 1e-4, require_ras: bool = True
) -> None:
    """Record an error when the named volumes disagree in grid or are not RAS."""
    infos = {}
    for n in names:
        p = findings.products.get(n)
        if p is not None:
            infos[n] = volume_info(Path(p.path))
    if not infos:
        return
    if require_ras:
        for n, i in infos.items():
            if i["axis_codes"] != ["R", "A", "S"]:
                findings.errors.append(
                    f"{n}: volume axes are {''.join(i['axis_codes'])}, expected RAS "
                    "(surface and volume frames must agree)"
                )
    ref_name = next(iter(infos))
    ref = infos[ref_name]
    for n, i in infos.items():
        if n == ref_name:
            continue
        if i["shape"][:3] != ref["shape"][:3]:
            findings.errors.append(
                f"{n}: grid {i['shape'][:3]} differs from {ref_name} grid {ref['shape'][:3]}"
            )
        elif not np.allclose(i["affine"], ref["affine"], atol=atol):
            findings.errors.append(f"{n}: affine differs from {ref_name} (max diff "
                                   f"{np.abs(i['affine'] - ref['affine']).max():.3g})")
    for n, i in infos.items():
        findings.products[n].meta.setdefault("shape", i["shape"][:3])
        findings.products[n].meta.setdefault("axis_codes", i["axis_codes"])
