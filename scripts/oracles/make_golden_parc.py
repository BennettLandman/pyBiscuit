#!/usr/bin/env python
"""Tiny committed golden for the parcellation net, made by the ORIGINAL code (plan row B7.2-oracle).

Run with the oracle venv:

    ~/dev/biscuit/oracles/parc-unet/venv/bin/python -I scripts/oracles/make_golden_parc.py

Builds the original ``model.SphericalUNet`` (staged, shimmed by ``parc_unet_oracle.install_shims``)
with ``in_ch=3, out_ch=5, max_level=3, min_level=0, fdim=8``, randomises every parameter and BN
statistic with a seeded generator (these are test weights, NOT the BISCUIT models), and stores the
state dict, a seeded input and the eval-mode output in ``tests/data/golden/parc-unet/tiny.npz``.
CI (no reference/ and no pickles) compares ``pybiscuit.parc`` against it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from parc_unet_oracle import install_shims  # noqa: E402

SEED = 20261020
STAGE = Path.home() / "dev/biscuit/oracles/parc-unet/stage"
OUT = Path(__file__).resolve().parents[2] / "tests/data/golden/parc-unet/tiny.npz"
ARCH = {"in_ch": 3, "out_ch": 5, "max_level": 3, "min_level": 0, "fdim": 8}


def main() -> None:
    install_shims(STAGE)
    import torch
    from cpuModel import cpuModel
    from model import SphericalUNet

    torch.manual_seed(SEED)
    net = SphericalUNet("./mesh_files", ARCH["in_ch"], ARCH["out_ch"], ARCH["max_level"],
                        ARCH["min_level"], ARCH["fdim"])  # fmt: skip
    model = cpuModel(net).eval()
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, t in model.state_dict().items():
            if t.is_sparse or name.endswith(".NS") or name.endswith(".EW") or name.endswith("num_batches_tracked"):
                continue
            if name.endswith("running_var"):
                t.copy_(0.5 + torch.rand(t.shape, generator=g))
            elif name.endswith("running_mean"):
                t.copy_(0.2 * torch.randn(t.shape, generator=g))
            elif name.endswith(".weight") and t.ndim == 1:  # BN scale
                t.copy_(0.8 + 0.4 * torch.rand(t.shape, generator=g))
            else:
                t.add_(0.05 * torch.randn(t.shape, generator=g))
        nv = 10 * 4 ** ARCH["max_level"] + 2
        x = torch.randn(1, ARCH["in_ch"], nv, generator=g)
        y = model(x)
    # sparse G/L/F2V buffers are operator data (the 2021 checkpoints do not contain them either)
    sd = {f"sd/{k}": v.numpy() for k, v in model.state_dict().items() if not v.is_sparse}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, x=x[0].numpy(), y=y[0].numpy(), arch=np.array(
        [ARCH[k] for k in ("in_ch", "out_ch", "max_level", "min_level", "fdim")]), **sd)  # fmt: skip
    print(f"{OUT}: {OUT.stat().st_size} B, {len(sd)} tensors, y range {y.min():.3f}..{y.max():.3f}")


if __name__ == "__main__":
    main()
