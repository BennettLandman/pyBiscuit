"""Command lines: ``pybiscuit mesh-contour`` (T39 stand-in) and ``pybiscuit hemi-split``."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

from pybiscuit.surfprep.contour import DEFAULT_RULE, PLANE_RULES, mesh_contour


def _read_surface(path) -> object:
    path = Path(path)
    if path.suffix == ".asc":
        from pybiscuit.io.freesurfer import read_asc

        return read_asc(path, frame="RAS")
    from pybiscuit.io.vtk import read_vtk

    return read_vtk(path)


def mesh_contour_main(argv: list[str]) -> int:
    """``MeshContour <mesh.vtk> nx ny nz d1,d2,...,`` as seen in spec T39.

    Prints the 0-based ids of the faces cut by the selected plane (one per
    line) and, last, the 0-based index of the selected offset, which is what
    ``hemi_separation.m`` parses with ``str2num``.  The selection rule is the
    documented hypothesis in :mod:`pybiscuit.surfprep.contour` (``--rule``).
    """
    # positional-only like the binary; offsets may start with '-', which argparse would
    # take for an option, so the (optional) --rule is pulled out by hand
    rule = DEFAULT_RULE
    args = list(argv)
    if "--rule" in args:
        i = args.index("--rule")
        rule = args[i + 1] if i + 1 < len(args) else ""
        del args[i : i + 2]
    if len(args) != 5 or rule not in PLANE_RULES:
        rules = "|".join(sorted(PLANE_RULES))
        sys.stderr.write(
            f"usage: pybiscuit mesh-contour <mesh.vtk> nx ny nz d1,d2,... [--rule {rules}]\n"
        )
        return 2
    a = argparse.Namespace(mesh=args[0], rule=rule)
    normal = [float(t) for t in args[1:4]]
    offsets = [float(t) for t in args[4].split(",") if t.strip()]
    if not offsets:
        sys.stderr.write("no plane offsets given\n")
        return 2
    mesh = _read_surface(a.mesh)
    ranking, cut = mesh_contour(mesh.vertices, mesh.faces, normal, offsets, a.rule)
    best = ranking[0]
    out = [str(int(i)) for i in cut[best]] + [str(best)]
    sys.stdout.write("\n".join(out) + "\n")
    return 0


def hemi_split_main(argv: list[str]) -> int:
    from pybiscuit.surfprep import frames
    from pybiscuit.surfprep.hemisphere import (
        HemisphereError,
        separate_hemispheres,
        write_outputs,
    )

    p = argparse.ArgumentParser(
        prog="pybiscuit hemi-split",
        description="Split the whole-brain MaCRUISE inner surface into closed hemispheres "
        "(port of hemi_separation.m). Alignment frame: --ref-surf, --aff or --seg "
        "(default: the native surface).",
    )
    p.add_argument("--surf", required=True, help="native inner surface (.asc scanner RAS, or .vtk)")
    p.add_argument("--out", required=True, help="output directory")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ref-surf", help="MNI copy with the same vertex order (.asc/.vtk)")
    g.add_argument("--aff", help="aff.txt (4x4)")
    g.add_argument("--seg", help="SEG_refine.nii.gz: fit the midline plane to the L/R labels")
    p.add_argument(
        "--aff-maps", default="mni_to_native", choices=["mni_to_native", "native_to_mni"]
    )
    p.add_argument("--rule", default=DEFAULT_RULE, choices=sorted(PLANE_RULES))
    p.add_argument("--max-plane-tries", type=int, default=5)
    p.add_argument("--strict-two-loops", action="store_true", help="fail on more than two loops")
    a = p.parse_args(argv)

    mesh = _read_surface(a.surf)
    if a.ref_surf:
        ref, source = (
            frames.reference_from_mni_vertices(_read_surface(a.ref_surf).vertices),
            "ref-surf",
        )
    elif a.aff:
        ref = frames.reference_from_affine(
            mesh.vertices, frames.read_aff(a.aff), aff_maps=a.aff_maps
        )
        source = "aff"
    elif a.seg:
        ref, source = frames.reference_from_labels(mesh.vertices, a.seg), "seg-labels"
    else:
        ref, source = None, "native"
    try:
        res = separate_hemispheres(
            mesh,
            ref,
            rule=a.rule,
            max_plane_tries=a.max_plane_tries,
            allow_extra_loops=not a.strict_two_loops,
        )
    except HemisphereError as exc:
        print(f"pybiscuit hemi-split: {exc}", file=sys.stderr)
        return 1
    paths = write_outputs(res, a.out)
    d = dataclasses.asdict(res.diagnostics)
    d.update(
        frame_source=source,
        n_input_vertices=int(mesh.n_vertices),
        n_mapped=int(res.hemi_map.shape[0]),
        n_lh=int(res.lh.n_vertices),
        n_rh=int(res.rh.n_vertices),
        status="provisional: no image oracle (Q1); MeshContour rule is hypothesis H-MC1",
    )
    (Path(a.out) / "hemisphere.json").write_text(json.dumps(d, indent=2, default=_json) + "\n")
    print(
        f"wrote {len(paths)} files to {a.out}; patch loops {d['loop_sizes']}, "
        f"lh {d['n_lh']} / rh {d['n_rh']} vertices"
    )
    return 0


def _json(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))
