"""pybiscuit.parc: cortical surface parcellation (plan rows B7.2–B7.5).

PyTorch is only required by the submodules that run or convert the network
(``unet``, ``checkpoint``, ``infer``); install it with ``pip install pybiscuit[parc]``.
Importing this package alone never imports torch or numba.  ``graphcut``, ``cleanup``,
``fusion``, and the legacy-pickle audit in ``checkpoint`` do not need torch.
"""

from __future__ import annotations

__all__ = [
    "CleanupInfo",
    "GraphCutInfo",
    "SizeTieEvent",
    "equal_label_components",
    "extract_parc_features",
    "fuse_folds",
    "gcut",
    "gcut_from_edges",
    "load_protocol_labels",
    "normalize_parc_feature",
    "parcellate",
    "refine_fold",
    "roi_correction",
]


def __getattr__(name: str):
    if name in ("GraphCutInfo", "gcut", "gcut_from_edges"):
        from . import graphcut

        return getattr(graphcut, name)
    if name in ("CleanupInfo", "SizeTieEvent", "equal_label_components", "roi_correction"):
        from . import cleanup

        return getattr(cleanup, name)
    if name in ("extract_parc_features", "load_protocol_labels", "normalize_parc_feature"):
        from . import features

        return getattr(features, name)
    if name in ("fuse_folds", "refine_fold"):
        from . import fusion

        return getattr(fusion, name)
    if name in ("parcellate",):
        from . import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
