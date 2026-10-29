"""pybiscuit.parc.pipeline: end-to-end cortical surface parcellation pipeline (B7.1, B7.3).

Mirrors ``parc.m`` (spec §6):
1. Extract and normalize parcellation features (``-H``, ``iH``, ``sulc``).
2. Truncated rigid spherical alignment via registration coefficients
   (``SphericalRemesh --deg <degree>``).
3. Spherical registration refinement against protocol template
   (``HSD -d 0 --noguess --icosahedron 5``).
4. Feature resampling onto ``icosphere_5`` (channel order ``[H, iH, sulc]``).
5. U-Net inference across model folds.
6. Resample raw class scores back to the aligned native sphere.
7. Refinement (``gcut`` + ``roi_correction``) and multi-model fusion (for DKT31).
8. Save ``<hemi>.parc.<protocol>.txt`` with protocol ROI IDs.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import numpy as np

from pybiscuit.assets import cache_dir
from pybiscuit.hsd.register import HSD
from pybiscuit.io import read_fs_curv, read_vertex_txt, read_vtk, write_vertex_txt
from pybiscuit.io.vtk import write_vtk
from pybiscuit.sphere import remesh
from pybiscuit.sphere.icosphere import icosphere

from .features import extract_parc_features, load_protocol_labels
from .fusion import fuse_folds

__all__ = [
    "find_protocol_assets",
    "parcellate",
]


def find_protocol_assets(
    protocol: str,
    hemi: str,
    *,
    assets_dir: str | Path | None = None,
) -> dict[str, Path | list[Path]]:
    """Locate protocol templates, labels, and checkpoint assets.

    Searches in:
    1. Provided ``assets_dir``
    2. Environment variable ``PYBISCUIT_ASSETS``
    3. Bundled/local reference directories
    4. ``~/.cache/pybiscuit/assets``
    """
    candidates = []
    if assets_dir:
        candidates.append(Path(assets_dir))
    if env := os.environ.get("PYBISCUIT_ASSETS"):
        candidates.append(Path(env))

    # Add standard project references
    repo_root = Path(__file__).resolve().parents[3]
    dev_root = repo_root.parent if (repo_root.parent / "reference").is_dir() else repo_root
    candidates.extend(
        [
            dev_root / "reference" / "converted",
            dev_root / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit",
            dev_root / "reference" / "ctf_plain" / "SurfPostproc" / "common",
            repo_root / "reference" / "converted",
            repo_root / "reference" / "sif_FC_v2.2" / "root" / "opt" / "biscuit",
            repo_root / "reference" / "ctf_plain" / "SurfPostproc" / "common",
            cache_dir(),
        ]
    )

    h = "lh" if hemi.lower().startswith("l") else "rh"
    proto_dir = None
    for c in candidates:
        candidate_proto = c / "parc" / protocol
        if candidate_proto.is_dir() and (candidate_proto / "label.csv").is_file():
            proto_dir = candidate_proto
            break

    if proto_dir is None:
        raise FileNotFoundError(
            f"Could not find assets for protocol {protocol!r} in any candidate location: "
            + ", ".join(str(c) for c in candidates)
        )

    label_csv = proto_dir / "label.csv"
    curv_mean = proto_dir / f"{h}.curv.mean.txt"
    if not curv_mean.is_file():
        raise FileNotFoundError(f"Template curvature file not found: {curv_mean}")

    # Find checkpoints: prefer safetensors, fall back to pth
    checkpoints: list[Path] = []
    # Check converted safetensors in candidate paths
    for c in candidates:
        converted_proto = c / "parc" / protocol
        sfts = sorted(converted_proto.glob(f"{h}*.safetensors"))
        if sfts:
            checkpoints = sfts
            break

    if not checkpoints:
        # Check .pth in proto_dir
        checkpoints = sorted(proto_dir.glob(f"{h}*.pth"))

    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found for {protocol} {h}")

    # Sort checkpoints naturally: lh.safetensors, lh_2.safetensors, ...
    def _ckpt_sort_key(p: Path) -> tuple[int, str]:
        m = re.search(r"_(\d+)\.", p.name)
        idx = int(m.group(1)) if m else 1
        return (idx, p.name)

    checkpoints.sort(key=_ckpt_sort_key)

    # Locate sphere_327680.vtk
    sphere_template = None
    for c in candidates:
        st = c / "sphere" / "sphere_327680.vtk"
        if st.is_file():
            sphere_template = st
            break

    # Locate icosphere_5.vtk
    ico5_mesh = None
    for c in candidates:
        for sub in ("mesh_files", "parc/mesh_files", "sphere"):
            candidate_ico = c / sub / "icosphere_5.vtk"
            if candidate_ico.is_file():
                ico5_mesh = candidate_ico
                break
        if ico5_mesh:
            break

    return {
        "proto_dir": proto_dir,
        "label_csv": label_csv,
        "curv_mean": curv_mean,
        "checkpoints": checkpoints,
        "sphere_template": sphere_template,
        "ico5_mesh": ico5_mesh,
    }


def parcellate(
    subject_dir: str | Path,
    hemi: str,
    protocol: str = "braincolor",
    out_dir: str | Path | None = None,
    *,
    degree: int = 0,
    ico_level: int = 5,
    assets_dir: str | Path | None = None,
    sphere_template: str | Path | None = None,
    robust_aabb: bool = False,
    quantize: bool = True,
    temp_dir: str | Path | None = None,
) -> np.ndarray:
    """Run cortical surface parcellation for a hemisphere and protocol.

    Parameters
    ----------
    subject_dir : str or Path
        Directory containing subject surfaces (e.g. ``<hemi>.white.vtk``, ``<hemi>.sphere.vtk``,
        ``<hemi>.coeff.txt``, ``<hemi>.inflated.H`` or ``.txt``, ``<hemi>.sulc`` or ``.txt``).
    hemi : str
        ``'lh'`` or ``'rh'``.
    protocol : str
        ``'braincolor'`` or ``'dkt31'``.
    out_dir : str or Path, optional
        Output directory to write ``<hemi>.parc.<protocol>.txt``.
    degree : int
        Degree for rigid spherical alignment (0 for MaCRUISE and FreeSurfer, 10 for CIVET).
    ico_level : int
        Icosahedron sampling level for network features (default 5, 10,242 vertices).
    assets_dir : str or Path, optional
        Path to BISCUIT assets directory.
    sphere_template : str or Path, optional
        Path to 163,842-vertex template uniform sphere (``sphere_327680.vtk``).
    robust_aabb : bool
        Opt-in float64 fallback in SphereAABB (Q16).
    quantize : bool
        Whether to emulate text-file float rounding (``fmt='%f'``) between stages.
    temp_dir : str or Path, optional
        Custom directory for intermediate files.

    Returns
    -------
    labels : np.ndarray
        1D int64 array of predicted protocol ROI IDs for each vertex.
    """
    sdir = Path(subject_dir)
    h = "lh" if hemi.lower().startswith("l") else "rh"

    # 1. Resolve subject input files
    def _find_input(name: str, subdirs: tuple[str, ...]) -> Path:
        if (sdir / name).is_file():
            return sdir / name
        for sub in subdirs:
            if (sdir / sub / name).is_file():
                return sdir / sub / name
        raise FileNotFoundError(f"Input file {name} not found in {sdir} or subdirs {subdirs}")

    white_path = _find_input(f"{h}.white.vtk", ("surf",))
    sphere_path = _find_input(f"{h}.sphere.vtk", ("surf",))
    coeff_path = _find_input(f"{h}.coeff.txt", ("reg", "surf"))

    # Inflated H
    try:
        ih_path = _find_input(f"{h}.inflated.H", ("surf",))
    except FileNotFoundError:
        ih_path = _find_input(f"{h}.inflated.H.txt", ("surf",))

    # Sulc
    try:
        sulc_path = _find_input(f"{h}.sulc", ("surf",))
    except FileNotFoundError:
        sulc_path = _find_input(f"{h}.sulc.txt", ("surf",))

    # Read meshes and scalar inputs
    white_mesh = read_vtk(white_path)

    # Read iH
    if ih_path.suffix == ".txt":
        ih_curv = read_vertex_txt(ih_path)
    else:
        ih_curv = read_fs_curv(ih_path)

    # Read sulc
    if sulc_path.suffix == ".txt":
        sulc_curv = read_vertex_txt(sulc_path)
    else:
        sulc_curv = read_fs_curv(sulc_path)

    # 2. Locate protocol assets
    assets = find_protocol_assets(protocol, h, assets_dir=assets_dir)
    label_ids, label_names = load_protocol_labels(assets["label_csv"], h)
    curv_mean_path = assets["curv_mean"]
    checkpoints = assets["checkpoints"]

    with tempfile.TemporaryDirectory(dir=temp_dir, prefix=f"parc_{protocol}_{h}_") as tmpdir:
        tmppath = Path(tmpdir)

        # Prepare icosphere_5
        if assets["ico5_mesh"] and Path(assets["ico5_mesh"]).is_file():
            ico5_vtk = Path(assets["ico5_mesh"])
            ico5 = read_vtk(ico5_vtk)
        else:
            ico5 = icosphere(ico_level, scheme="ugscnn")
            ico5_vtk = tmppath / f"icosphere_{ico_level}.vtk"
            write_vtk(ico5, ico5_vtk)

        # Prepare sphere_327680
        if sphere_template and Path(sphere_template).is_file():
            tmpl_sphere_path = Path(sphere_template)
        elif assets["sphere_template"] and Path(assets["sphere_template"]).is_file():
            tmpl_sphere_path = Path(assets["sphere_template"])
        else:
            ico7 = icosphere(7, scheme="biscuit")
            tmpl_sphere_path = tmppath / "sphere_327680.vtk"
            write_vtk(ico7, tmpl_sphere_path)

        # 3. Extract and normalise features
        h_norm, ih_norm, sulc_norm = extract_parc_features(
            white_mesh,
            ih_curv,
            sulc_curv,
            round_intermediate=quantize,
        )

        h_feat_txt = tmppath / f"{h}.H.txt"
        ih_feat_txt = tmppath / f"{h}.iH.txt"
        sulc_feat_txt = tmppath / f"{h}.sulc.txt"
        write_vertex_txt(h_norm, h_feat_txt, fmt="%f")
        write_vertex_txt(ih_norm, ih_feat_txt, fmt="%f")
        write_vertex_txt(sulc_norm, sulc_feat_txt, fmt="%f")

        # 4. Spherical alignment (spec §6.3 T27, T28)
        # T27: Truncated rigid deformation by registration coefficients
        aligned_sphere_path = tmppath / f"{h}.aligned.sphere.vtk"
        remesh.run(
            sphere=sphere_path,
            coeff=coeff_path,
            ref=ico5_vtk,
            deg=degree,
            deform=aligned_sphere_path,
            quiet=True,
        )

        # T28: Rigid refinement against protocol template via HSD
        hsd = HSD(
            [str(tmpl_sphere_path), str(aligned_sphere_path)],
            [[str(curv_mean_path)], [str(h_feat_txt)]],
            output=[str(aligned_sphere_path), str(aligned_sphere_path)],
            degree=0,
            fixed_subjects=[0],
            guess=False,
            icosahedron_level=ico_level,
            robust_aabb=robust_aabb,
        )
        hsd.run()

        aligned_sphere_mesh = read_vtk(aligned_sphere_path)

        # 5. T29: ``SphericalRemesh -r icosphere_5 -s <tmp>.sphere.vtk -p <tmp>.{iH,sulc,H}.txt``
        # on the feature files as written (``%f``)
        feats = [ih_feat_txt, sulc_feat_txt, h_feat_txt]
        vals = [read_vertex_txt(f) for f in feats] if quantize else [ih_norm, sulc_norm, h_norm]
        res_iH, res_sulc, res_H = remesh.resample(aligned_sphere_path, ico5_vtk, vals)

        if quantize:
            res_H = np.round(res_H, 6)
            res_iH = np.round(res_iH, 6)
            res_sulc = np.round(res_sulc, 6)

        # Network input channel order is [H, iH, sulc] (spec §6.4)
        net_features = np.stack([res_H, res_iH, res_sulc], axis=0).astype(np.float32)

        # 6. Model inference across folds
        from .infer import ParcModel, resample_to_native

        fold_scores_list: list[np.ndarray] = []
        for ckpt in checkpoints:
            model = ParcModel.from_checkpoint(ckpt)
            raw_scores = model.predict(net_features)  # (nclass, 10242)

            # Resample raw class scores back to the aligned native sphere
            native_scores = resample_to_native(
                raw_scores,
                ico5,
                aligned_sphere_mesh,
                quantize=quantize,
            )  # (nclass, nV)
            fold_scores_list.append(native_scores)

        # 7. Refine and fuse folds
        final_labels = fuse_folds(
            white_mesh.vertices,
            white_mesh.faces,
            fold_scores_list,
            labels=label_ids,
        )

    # 8. Save output if requested
    if out_dir is not None:
        out_path = Path(out_dir) / f"{h}.parc.{protocol}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_vertex_txt(final_labels, out_path, fmt="%d")

    return final_labels
