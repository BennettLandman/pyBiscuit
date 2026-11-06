"""Shape Complexity Index: port of ``ShapeComplexityIndex`` (B9.6, orchestrator spec §9.4).

Computes the local shape complexity using 1D Earth Mover's Distance (EMD) on a histogram of
shape index values within a geodesic neighbourhood.

Reference implementation:
    ``NIRALUser/ShapeComplexityIndex`` (Sun Hyung Kim and Martin Styner, UNC Chapel Hill).
    Algorithm:
    1. Smooth the input surface with 2 iterations of Laplacian (nearest-neighbour mean) smoothing
       (``SurfaceUtil::smoothing(mesh, 2)``).
    2. Compute principal curvatures (``cmin``, ``cmax``) on the smoothed mesh using MeshLib's
       ``SurfaceUtil::curvature`` (double-precision tensor and eigensolve, cast to float32).
    3. Compute shape index:
       ``SI = (2 / pi) * atan((cmax + cmin) / (cmax - cmin))``, clamped to ``[-1.0, 1.0]``.
    4. For each vertex ``i``:
       - Propagate fast-marching geodesic front up to radius ``dmax = kernel_size`` (3.0 mm).
       - Collect all neighbour vertices ``j != i`` with ``dist[j] <= dmax``, ordered by index ``j``.
       - Let ``test_n = len(nbrs) - 1``. If ``test_n <= 0``, ``sci = 1.0``.
       - Shift values by +1 (range ``[0, 2]``) and bin into ``2^step + 1`` buckets
         (default ``step=5`` -> 33 bins of width 0.0625).
       - Compute 1D Earth Mover's Distance against each basis delta distribution ``Basis_X[b]``:
         ``EMD(b) = sum_k (hist[k] * |Basis_X[k] - Basis_X[b]|) / test_n``.
       - Take ``min_emd = min(0.5, min_b EMD(b))`` (upstream ``Minimum``: ``temp = 0.5f``).
       - ``SCI[i] = 2.0 * min_emd``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit

from ..io.surface import read_surface
from ..mesh.curvature import curvature
from ..mesh.geodesic import GeodesicSolver, _march, _reset
from ..mesh.geometry import smooth_vertices
from ..mesh.model import Mesh

__all__ = [
    "compute_shape_index",
    "shape_complexity_index",
    "write_sci",
]


def compute_shape_index(cmin: np.ndarray, cmax: np.ndarray) -> np.ndarray:
    """Compute shape index from principal curvatures as in ``CCalculators::Cal_ShapeIndex``.

    ``SI = (2 / pi) * atan((cmax + cmin) / (cmax - cmin))`` clamped to ``[-1.0, 1.0]``.
    Ties (``cmax == cmin``) are resolved as:
      - cmin > 0 and cmax > 0 -> 1.0
      - cmin < 0 and cmax < 0 -> -1.0
      - cmin == 0 and cmax == 0 -> 1.0
    """
    cmin = np.asarray(cmin, dtype=np.float32)
    cmax = np.asarray(cmax, dtype=np.float32)
    diff = cmax - cmin
    sum_c = cmax + cmin
    si = np.empty_like(cmin)

    mask_eq = diff == 0.0
    si[mask_eq & (cmin > 0.0)] = 1.0
    si[mask_eq & (cmin < 0.0)] = -1.0
    si[mask_eq & (cmin == 0.0)] = 1.0

    mask_neq = ~mask_eq
    # Upstream uses #define PI 3.14159265
    inv_half_pi = np.float32(2.0 / 3.14159265)
    si[mask_neq] = inv_half_pi * np.arctan(sum_c[mask_neq] / diff[mask_neq])
    np.clip(si, -1.0, 1.0, out=si)
    return si


@njit(cache=True)
def _compute_sci_loop(
    pos: np.ndarray,
    vert: np.ndarray,
    fnbr: np.ndarray,
    vptr: np.ndarray,
    vidx: np.ndarray,
    fptr: np.ndarray,
    fidx: np.ndarray,
    mode: int,
    si: np.ndarray,
    dmax: float,
    step: int,
) -> np.ndarray:
    n = pos.shape[0]
    n_basic = (1 << step) + 1  # 2^step + 1
    step_size = np.float32(1.0 / (1 << (step - 1)))  # 1 / 2^(step-1)

    basis_x = np.empty(n_basic, dtype=np.float32)
    for k in range(n_basic):
        basis_x[k] = np.float32(-1.0) + np.float32(k) * step_size

    cost_matrix = np.empty((n_basic, n_basic), dtype=np.float32)
    for r in range(n_basic):
        for c in range(n_basic):
            cost_matrix[r, c] = abs(basis_x[r] - basis_x[c])

    dist = np.full(n, 1e9, dtype=np.float64)
    state = np.zeros(n, dtype=np.int8)
    source = np.full(n, -1, dtype=np.int32)
    heap = np.empty(n + 1, dtype=np.int32)
    touched = np.empty(n + 1, dtype=np.int32)
    nt = 0

    starts = np.empty(1, dtype=np.int32)
    ends = np.empty(0, dtype=np.int32)
    niter = int(n * 1.2)

    nbrs_buf = np.empty(1024, dtype=np.int32)
    hist = np.zeros(n_basic, dtype=np.float32)
    sci_out = np.empty(n, dtype=np.float32)

    for i in range(n):
        _reset(dist, state, source, touched, nt)
        starts[0] = i
        nt = _march(
            pos,
            vert,
            fnbr,
            vptr,
            vidx,
            fptr,
            fidx,
            starts,
            ends,
            dmax,
            niter,
            mode,
            dist,
            state,
            source,
            heap,
            touched,
        )

        cnt = 0
        for k in range(nt):
            u = touched[k]
            if u != i and dist[u] <= dmax:
                if cnt < len(nbrs_buf):
                    nbrs_buf[cnt] = u
                    cnt += 1

        # Upstream C++ iterates for (int j = 0; j < n_vertex; j++) if (dist[j] <= dmax && j != i)
        # which visits candidate vertices in ascending vertex index order.
        nbrs = nbrs_buf[:cnt]
        nbrs.sort()

        # Upstream: int Test_N = Testing_SI.size() - 1;
        test_n = cnt - 1
        if test_n <= 0:
            # When Test_N <= 0, upstream Minimum returns initial temp = 0.5f, 2 * 0.5f = 1.0f
            sci_out[i] = 1.0
            continue

        hist[:] = 0.0
        for k in range(test_n):
            u = nbrs[k]
            val = si[u] + np.float32(1.0)
            b = int(val / step_size)
            if b < 0:
                b = 0
            elif b >= n_basic:
                b = n_basic - 1
            hist[b] += 1.0

        # Minimum EMD across basis bins; upstream Minimum initializes temp = 0.5f
        min_emd = np.float32(0.5)
        inv_tn = np.float32(1.0) / np.float32(test_n)
        for b in range(n_basic):
            s = np.float32(0.0)
            for k in range(n_basic):
                s += hist[k] * cost_matrix[k, b]
            emd = s * inv_tn
            if emd < min_emd:
                min_emd = emd

        sci_out[i] = np.float32(2.0) * min_emd

    return sci_out


def shape_complexity_index(
    mesh_or_path: Mesh | str | Path,
    *,
    kernel_size: float = 3.0,
    step: int = 5,
    heap: str = "libstdcxx",
) -> np.ndarray:
    """Compute Shape Complexity Index on a surface mesh.

    Parameters
    ----------
    mesh_or_path:
        Mesh object or path to legacy VTK polydata triangle mesh.
    kernel_size:
        Geodesic radius in mm (default 3.0, matching BISCUIT ``sci_kernel_size``).
    step:
        Exponent controlling number of histogram bins: ``2^step + 1`` bins
        (default 5 -> 33 bins, matching BISCUIT ``sci_bins``).
    heap:
        Geodesic solver heap implementation (default ``"libstdcxx"`` matching the Linux image).

    Returns
    -------
    np.ndarray:
        float32 array of shape (N,) containing per-vertex Shape Complexity Index in [0, 1].
    """
    mesh = read_surface(mesh_or_path) if not isinstance(mesh_or_path, Mesh) else mesh_or_path

    # Step 1: 2 iterations of Laplacian (NN mean) smoothing
    smoothed_v = smooth_vertices(mesh.vertices, faces=mesh.faces, iterations=2)
    smoothed_mesh = Mesh(smoothed_v, mesh.faces)

    # Step 2: Principal curvatures (float32 overload)
    curv = curvature(smoothed_mesh, dtype=np.float32)

    # Step 3: Shape Index
    si = compute_shape_index(curv.cmin, curv.cmax)

    # Step 4: Geodesic solver & fast marching
    geo = GeodesicSolver(smoothed_mesh, heap=heap)

    # Step 5: Fast marching + EMD calculation
    return _compute_sci_loop(
        geo.pos,
        geo.vert,
        geo.fnbr,
        geo.vptr,
        geo.vidx,
        geo.fptr,
        geo.fidx,
        geo.mode,
        si,
        float(kernel_size),
        int(step),
    )


def write_sci(sci: np.ndarray, path: str | Path) -> None:
    """Write SCI scalar map to text file (%f format, one per line)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for val in sci:
            f.write(f"{val:.6f}\n")
