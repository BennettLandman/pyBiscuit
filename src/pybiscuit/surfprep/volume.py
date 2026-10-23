"""Volume preparation and synthesis for MaCRUISE surface prep (plan B5.3; spec §4.3, §4.5).

Synthesizes the auxiliary FreeSurfer-style volumes from normalised T1 and
``SEG_refine.nii.gz`` (BrainCOLOR labels):

* ``brain.finalsurfs``: T1 masked by ``seg > 0``;
* ``wm``: white matter intensity mask (spec §4.5 table);
* ``filled``: hemisphere filling mask (spec §4.5 table: 255 = left, 127 = right);
* ``aseg``: BrainCOLOR labels re-mapped to FreeSurfer aseg IDs (spec §4.5 table).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np

# Spec §4.5 aseg replacements (evaluated on the original segmentation, no chaining)
BRAINCOLOR_TO_ASEG: dict[int, int] = {
    45: 2,  # Left-Cerebral-White-Matter
    44: 41,  # Right-Cerebral-White-Matter
    58: 12,  # Left-Putamen
    56: 13,  # Left-Pallidum
    57: 51,  # Right-Putamen
    55: 52,  # Right-Pallidum
    50: 5,  # Left-Lateral-Ventricle
    49: 44,  # Right-Lateral-Ventricle
    52: 4,  # Left-Lateral-Ventricle (inferior)
    51: 43,  # Right-Lateral-Ventricle (inferior)
    60: 10,  # Left-Thalamus-Proper
    59: 49,  # Right-Thalamus-Proper
    62: 28,  # Left-VentralDC
    61: 60,  # Right-VentralDC
    48: 17,  # Left-Hippocampus
    47: 53,  # Right-Hippocampus
    36: 50,  # Right-Caudate
    37: 11,  # Left-Caudate
    31: 54,  # Right-Amygdala
    32: 18,  # Left-Amygdala
    39: 8,  # Left-Cerebellum-Cortex
    38: 47,  # Right-Cerebellum-Cortex
    41: 7,  # Left-Cerebellum-White-Matter
    40: 46,  # Right-Cerebellum-White-Matter
    23: 58,  # Right-Accumbens-area
    30: 26,  # Left-Accumbens-area
    35: 16,  # Brain-Stem
    11: 15,  # 4th-Ventricle
    71: 47,  # Cerebellum
    72: 47,  # Cerebellum
    73: 47,  # Cerebellum
}

# Spec §4.5 filled tables (255 = left, 127 = right)
FILLED_LEFT_LABELS: tuple[int, ...] = (37, 45, 52, 56, 58, 60, 62)
FILLED_RIGHT_LABELS: tuple[int, ...] = (36, 44, 51, 55, 57, 59, 61)

# Spec §4.5 wm tables:
# Set to 0 wherever seg not in {44, 45, 35}
# Set to 250 wherever seg in {36, 37, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62}
WM_KEEP_T1_LABELS: tuple[int, ...] = (44, 45, 35)
WM_FIXED_250_LABELS: tuple[int, ...] = (36, 37, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62)


def synthesize_aseg(seg_data: np.ndarray) -> np.ndarray:
    """Map BrainCOLOR / SEG_refine labels to FreeSurfer aseg IDs (spec §4.5).

    All replacements are evaluated on the input array without chaining:
    - Specific subcortical / ventricular / cerebellar label mappings;
    - Cortex: every label >= 100 with odd value -> 3 (left cortex),
      even value -> 42 (right cortex).
    """
    seg = np.asarray(seg_data, dtype=np.int32)
    out = np.zeros_like(seg, dtype=np.int32)

    # 1. Cortex rule: labels >= 100
    cortex_mask = seg >= 100
    left_cortex = cortex_mask & (seg % 2 == 1)
    right_cortex = cortex_mask & (seg % 2 == 0)
    out[left_cortex] = 3
    out[right_cortex] = 42

    # 2. Specific label mappings
    for src_val, dst_val in BRAINCOLOR_TO_ASEG.items():
        out[seg == src_val] = dst_val

    return out


def synthesize_filled(seg_data: np.ndarray) -> np.ndarray:
    """Synthesize FreeSurfer-style ``filled`` volume from SEG_refine (spec §4.5).

    Returns uint8 array with:
    - 255 for left hemisphere structures (37, 45, 52, 56, 58, 60, 62)
    - 127 for right hemisphere structures (36, 44, 51, 55, 57, 59, 61)
    - 0 elsewhere
    """
    seg = np.asarray(seg_data, dtype=np.int32)
    out = np.zeros(seg.shape, dtype=np.uint8)

    left_mask = np.isin(seg, FILLED_LEFT_LABELS)
    right_mask = np.isin(seg, FILLED_RIGHT_LABELS)

    out[left_mask] = 255
    out[right_mask] = 127
    return out


def synthesize_wm(t1_data: np.ndarray, seg_data: np.ndarray) -> np.ndarray:
    """Synthesize FreeSurfer-style ``wm`` volume from T1 and SEG_refine (spec §4.5).

    Starts from T1 intensity (clamped to uint8 0..255):
    - set to 0 wherever seg not in {44, 45, 35};
    - set to 250 wherever seg in {36, 37, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62}.
    """
    t1 = np.clip(np.asarray(t1_data, dtype=np.float32), 0, 255).astype(np.uint8)
    seg = np.asarray(seg_data, dtype=np.int32)

    out = np.zeros_like(t1, dtype=np.uint8)

    # Keep T1 where seg in {44, 45, 35}
    keep_mask = np.isin(seg, WM_KEEP_T1_LABELS)
    out[keep_mask] = t1[keep_mask]

    # Set to 250 where seg in WM_FIXED_250_LABELS
    fixed_mask = np.isin(seg, WM_FIXED_250_LABELS)
    out[fixed_mask] = 250

    return out


def synthesize_brain_finalsurfs(t1_data: np.ndarray, seg_data: np.ndarray) -> np.ndarray:
    """Synthesize ``brain.finalsurfs``: T1 masked by ``seg > 0`` (spec §4.5)."""
    t1 = np.clip(np.asarray(t1_data, dtype=np.float32), 0, 255).astype(np.uint8)
    seg = np.asarray(seg_data, dtype=np.int32)

    out = np.zeros_like(t1, dtype=np.uint8)
    brain_mask = seg > 0
    out[brain_mask] = t1[brain_mask]
    return out


def conform_volume(
    img: nib.spatialimages.SpatialImage,
    is_label: bool = False,
    out_shape: tuple[int, int, int] = (256, 256, 256),
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    orientation: str = "LIA",
) -> nib.spatialimages.SpatialImage:
    """Conform volume to standard shape and resolution (spec §4.3)."""
    order = 0 if is_label else 3
    return nibproc.conform(
        img,
        out_shape=out_shape,
        voxel_size=voxel_size,
        order=order,
        orientation=orientation,
    )


def prepare_auxiliary_volumes(
    t1_img: nib.spatialimages.SpatialImage,
    seg_img: nib.spatialimages.SpatialImage,
    out_dir: str | Path,
    conform: bool = True,
) -> dict[str, Path]:
    """Generate all auxiliary volumes required for surface prep and write to out_dir.

    Generates:
    - ``brain.finalsurfs.mgz``
    - ``wm.mgz``
    - ``filled.mgz``
    - ``aseg.mgz``

    Returns a dict mapping volume name to Path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if conform:
        t1_c = conform_volume(t1_img, is_label=False)
        seg_c = conform_volume(seg_img, is_label=True)
    else:
        t1_c = t1_img
        seg_c = seg_img

    t1_data = np.asarray(t1_c.dataobj)
    seg_data = np.asarray(seg_c.dataobj)
    affine = t1_c.affine
    header = t1_c.header

    results: dict[str, Path] = {}

    vols = {
        "brain.finalsurfs": (synthesize_brain_finalsurfs(t1_data, seg_data), np.uint8),
        "wm": (synthesize_wm(t1_data, seg_data), np.uint8),
        "filled": (synthesize_filled(seg_data), np.uint8),
        "aseg": (synthesize_aseg(seg_data), np.int32),
    }

    for name, (data, dtype) in vols.items():
        p = out_dir / f"{name}.mgz"
        # Save as MGH format with conformed affine
        img = nib.MGHImage(data.astype(dtype), affine, header)
        nib.save(img, str(p))
        results[name] = p

    return results
