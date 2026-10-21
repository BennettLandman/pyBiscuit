#!/usr/bin/env python
"""Run the image's original parcellation ``test.py`` under current PyTorch (plan row B7.2-oracle).

The oracle is the plaintext Python shipped in the BISCUIT 2.2 image CTF
(``SurfPostproc/common/parc/{test,model,loader_individual,cpuModel}.py``, ``meshcnn/*.py``,
``mesh_files/icosphere_{0..5}.pkl``), copied byte-for-byte into a staging directory and executed
unchanged with ``runpy``.  Only *import-level shims* are applied, all in this file:

1. ``ops.pickle`` -> :class:`_ShimPickle`.  ``ops.py`` calls ``pickle.load`` on the shipped
   ``icosphere_*.pkl`` files, which are third-party Python-2-era pickles: never unpickled with the
   stock loader.  The shim reads them with a restricted unpickler that allows exactly the six
   globals seen with ``pickletools`` (numpy ``_reconstruct``/``ndarray``/``dtype``, scipy
   ``csc_matrix``/``coo_matrix``, ``_codecs.encode``) and maps them to current classes.
2. ``scipy.sparse.coo`` alias (``utils.sparse2tensor`` asserts ``isinstance(m,
   sparse.coo.coo_matrix)``; the ``coo`` submodule is gone in current SciPy).
3. ``torch.sparse.FloatTensor`` (used by ``utils.sparse2tensor``) -> ``torch.sparse_coo_tensor``
   when the legacy constructor is gone.
4. ``np.savetxt`` also saves the float32 array as ``<file>.npy`` (the text file has 6 decimals
   only; the npy is the exact network output).
5. ``os.system`` copies the per-class text files into ``raw/`` before running ``SphericalRemesh``
   (which overwrites them), then runs the command unchanged with the oracle binary on ``PATH``.

``torch.load`` in ``test.py`` is untouched; current torch defaults to ``weights_only=True``, which
is what we want for third-party checkpoints.

Run with the oracle venv's python (torch, numpy, scipy only), e.g.::

    parc_unet_oracle.py --stage ~/dev/biscuit/oracles/parc-unet/stage --ckpt lh.pth --out-ch 36 \\
        --features feats.npy --workdir /tmp/run1 --isphere icosphere_5.vtk --osphere native.vtk
"""

from __future__ import annotations

import argparse
import codecs
import os
import pickle
import runpy
import shutil
import sys
import types
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- shim 1: restricted pickle


class _ShimPickle(types.ModuleType):
    """Stands in for the ``pickle`` module inside ``ops``; only ``load`` is used there."""

    _ALLOWED = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("scipy.sparse.csc", "csc_matrix"),
        ("scipy.sparse.coo", "coo_matrix"),
        ("_codecs", "encode"),
    }

    def __init__(self):
        super().__init__("pickle_shim")

    def load(self, fh, *a, **k):
        allowed = self._ALLOWED

        class _U(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) not in allowed:
                    raise pickle.UnpicklingError(f"blocked global {module}.{name}")
                if module == "numpy.core.multiarray":
                    from numpy._core.multiarray import _reconstruct

                    return _reconstruct
                if module.startswith("scipy.sparse"):
                    import scipy.sparse as sp

                    return getattr(sp, name)
                if module == "_codecs":

                    def enc(obj, encoding="utf-8", errors="strict"):
                        if encoding != "latin1" or not isinstance(obj, str):
                            raise pickle.UnpicklingError("blocked _codecs.encode use")
                        return codecs.encode(obj, encoding, errors)

                    return enc
                return getattr(__import__(module, fromlist=[name]), name)

            def persistent_load(self, pid):
                raise pickle.UnpicklingError("persistent ids refused")

        return _U(fh, encoding="latin1").load()


# ---------------------------------------------------------------- main


def install_shims(stage: Path, work: Path | None = None, remesh_bin: str | None = None):
    """Prepare ``sys.path``/cwd for the staged original code and apply the shims above.

    Returns the (patched) ``ops`` module.  ``work`` enables the ``os.system`` recorder (shim 5).
    """
    os.chdir(stage)
    sys.path.insert(0, str(stage))
    sys.path.insert(0, str(stage / "meshcnn"))

    import scipy.sparse as sp
    import torch

    if not hasattr(sp, "coo"):  # shim 2
        sp.coo = types.SimpleNamespace(coo_matrix=sp.coo_matrix)
    if not hasattr(torch.sparse, "FloatTensor"):  # shim 3
        torch.sparse.FloatTensor = lambda i, v, size: torch.sparse_coo_tensor(i, v, size)

    import ops  # original meshcnn/ops.py

    ops.pickle = _ShimPickle()  # shim 1

    real_savetxt = np.savetxt

    def savetxt(fname, X, *args, **kw):  # shim 4
        real_savetxt(fname, X, *args, **kw)
        np.save(f"{fname}.npy", np.asarray(X, dtype=np.float32))

    np.savetxt = savetxt

    if work is not None:
        real_system = os.system

        def system(cmd):  # shim 5
            toks = cmd.split()
            if "-p" in toks:
                i = toks.index("-p") + 1
                while i < len(toks) and not toks[i].startswith("-"):
                    p = Path(toks[i])
                    shutil.copy(p, work / "raw" / p.name)
                    shutil.copy(f"{p}.npy", work / "raw" / f"{p.name}.npy")
                    i += 1
            if remesh_bin:
                os.environ["PATH"] = f"{remesh_bin}:{os.environ['PATH']}"
            return real_system(cmd)

        os.system = system
    return ops


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, help="dir with the original parc/ tree")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-ch", type=int, required=True)
    ap.add_argument("--features", required=True, help=".npy float array (>=3, nv) = H, iH, sulc")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--max-level", type=int, default=5)
    ap.add_argument("--isphere", help="icosahedral sphere (SphericalRemesh -s)")
    ap.add_argument("--osphere", help="native sphere (SphericalRemesh -r)")
    ap.add_argument("--remesh-bin", help="dir containing the SphericalRemesh binary")
    a = ap.parse_args()

    stage = Path(a.stage).resolve()
    work = Path(a.workdir).resolve()
    if work.exists():
        shutil.rmtree(work)
    (work / "raw").mkdir(parents=True)
    feats = np.load(a.features)
    for name, row in zip(("H", "iH", "sulc"), feats[:3], strict=True):
        np.savetxt(work / f"subj.res.{name}.txt", row)

    install_shims(stage, work, a.remesh_bin)
    sys.argv = [
        "test.py",
        "--ckpt", a.ckpt,
        "--export_file", "out.mat",
        "--test-batch-size", "1",
        "--data_folder", str(work),
        "--max_level", str(a.max_level),
        "--min_level", "0",
        "--feat", "64",
        "--in_ch", "3",
        "--out_ch", str(a.out_ch),
        "--isphere", a.isphere or "none",
        "--osphere", a.osphere or "none",
        "--no-cuda",
    ]  # fmt: skip
    runpy.run_path("test.py", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
