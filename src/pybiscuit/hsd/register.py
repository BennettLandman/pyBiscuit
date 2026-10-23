# ruff: noqa: E501
"""HSD spherical registration: port of ``HSD`` (ilwoolyu/HSD @ f169595, ``src/HSD.cpp``).

Property-only registration as reachable from the shipped CLI (no landmarks / location terms;
see ``docs/ports/hsd.md``).  Method names follow the C++ (snake_case).  State is kept in NumPy
arrays with the C++ dtypes: sphere vertices float32, SH basis / coefficients / gradients float64,
features float32, mean / variance float64.
"""

from __future__ import annotations

import math
import sys

import numpy as np

from pybiscuit.mesh.geometry import face_normals, smooth_data, vertex_neighbor_lists
from pybiscuit.sphere.aabb import SphereAABB
from pybiscuit.sphere.remesh import _load_mesh, _read_floats, _write_vtk_meshlib
from pybiscuit.sphere.spharm import (
    ShDeformation,
    basis,
    read_sh_coefficients,
    write_sh_coefficients,
)

from . import gradient as G
from . import kernels as K
from .geom import F, unit_rows_inplace
from .guess import guess_init_coeff
from .icosahedron import icosahedron

FLT_MAX = 3.4028234663852886e38
_SAMPLES = {642: 3, 2562: 4, 10242: 5, 40962: 6, 163842: 7}


class _Subject:
    pass


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(v, dtype=np.float32).copy()
    unit_rows_inplace(v)
    return v


def _tree(verts, faces, normals_from):
    """AABB_Sphere on ``verts`` with pole detection using the face normals of ``normals_from``
    (``Mesh`` computes normals when the mesh is built; ``setVertex`` never refreshes them)."""
    t = SphereAABB(verts, faces)
    t._fnormals = face_normals(normals_from, faces)
    t._refresh()
    return t


def _patch_cache(faces: np.ndarray) -> np.ndarray:
    """``initLocalPatch``: neighbouring face across each edge (first inserted directed edge wins)."""
    nf = len(faces)
    nv = int(faces.max()) + 1
    f = faces.astype(np.int64)
    d = np.stack([f[:, 0] * nv + f[:, 1], f[:, 1] * nv + f[:, 2], f[:, 2] * nv + f[:, 0]], 1)
    keys, first = np.unique(d.ravel(), return_index=True)
    owner = first // 3
    q = np.stack([f[:, 1] * nv + f[:, 0], f[:, 2] * nv + f[:, 1], f[:, 0] * nv + f[:, 2]], 1)
    pos = np.searchsorted(keys, q.ravel())
    pos = np.minimum(pos, len(keys) - 1)
    hit = keys[pos] == q.ravel()
    out = np.where(hit, owner[pos], -1).astype(np.int32)
    return np.ascontiguousarray(out.reshape(nf, 3))


class HSD:
    """Constructor arguments mirror the CLI (defaults are the CLI's, not the C++ constructor's)."""

    def __init__(
        self,
        spheres,
        properties,
        *,
        output=None,
        output_coeff=None,
        weight=None,
        degree=15,
        idprior=625.0,
        coefficient=None,
        max_iter=20,
        fixed_subjects=None,
        icosahedron_level=7,
        realtime_coeff=False,
        tmp_variance=None,
        guess=True,
        guess_res=3,
        ico_mesh=None,
        resampling=False,
        guess_unit_ico=True,
        verbose=True,
    ):
        self.verbose = verbose
        self.log: list[str] = []
        self.history: list[tuple[int, float]] = []
        self.nsubj = len(spheres)
        self.nprop = len(properties[0])
        if any(len(p) != self.nprop for p in properties) or len(properties) != self.nsubj:
            raise ValueError("every subject needs the same number of property files")
        self.weight = np.ones(self.nprop, np.float32) if weight is None else np.asarray(weight, np.float32)
        self.max_iter = int(max_iter)
        self.degree = int(degree)
        self.K = (self.degree + 1) ** 2
        self.lambda1 = 1.0
        self.lambda2 = float(F(idprior))
        self.eta = 1.0
        self.fine_res = 6
        self.guess = bool(guess)
        self.guess_active = bool(guess)
        self.guess_res = int(guess_res)
        self.guess_unit_ico = bool(guess_unit_ico)
        self.degree_inc = 0
        self.realtime_coeff = bool(realtime_coeff)
        self.icosahedron = int(icosahedron_level)
        self.resampling = bool(resampling)
        self.output = output
        self.output_coeff = output_coeff
        self.mincost = np.float32(FLT_MAX)
        self.n_iter = 0
        self.n_success = 0
        self.cost_prop = self.cost_area = self.cost_disp = 0.0
        self.coeff = np.zeros((self.nsubj, 3, self.K))
        self.coeff_prev = np.zeros_like(self.coeff)
        self.gradient = np.zeros_like(self.coeff)
        self.updated = np.zeros(self.nsubj, bool)
        fixed = np.zeros(self.nsubj, bool)
        for i in fixed_subjects or []:
            fixed[i] = True
        self.fixed = fixed
        self.ndef = int((~fixed).sum())
        self._init(spheres, properties, coefficient, ico_mesh, tmp_variance)

    # ------------------------------------------------------------------ utils
    def _say(self, msg: str = "") -> None:
        self.log.append(msg)
        if self.verbose:
            print(msg)
            sys.stdout.flush()

    # ------------------------------------------------------------------- init
    def _init(self, spheres, properties, coefficient, ico_mesh, tmp_variance):
        nsubj = self.nsubj
        # icosahedron used for evaluation (uniform sampling points)
        if ico_mesh is None:
            self.ico_v, self.ico_f = icosahedron(self.icosahedron)
            self.ico_v_raw = self.ico_v.copy()
            self.samples = self.ico_v.copy()
        else:
            self.ico_v, self.ico_f = _load_mesh(ico_mesh)
            n = len(self.ico_v)
            if n not in _SAMPLES:
                raise ValueError(f"icosahedron mesh is invalid (vert#={n})")
            self.icosahedron = _SAMPLES[n]
            self.ico_v_raw = self.ico_v.copy()
            self.samples = _unit(self.ico_v)
            self._say(f"Icosahedron level: {self.icosahedron}")
        self.nsamples = len(self.samples)
        self.pairwise = self.ndef == 1 and nsubj == 2
        if self.pairwise:
            self._say("Running on pair-wise registration")
        ico_Y = None
        if self.resampling:
            self.ico_v = _unit(self.ico_v)  # normalised in place upstream
            ico_Y = basis(self.ico_v.astype(np.float64), self.degree)
        self._ring_cache: dict[int, tuple] = {}
        self.subj: list[_Subject] = []
        self._say("Initialzation of subject information")
        for s in range(nsubj):
            sub = _Subject()
            sub.fixed = bool(self.fixed[s])
            raw, faces0 = _load_mesh(spheres[s])
            v0 = _unit(raw)
            tree0 = _tree(v0, faces0, raw)
            resample_here = self.resampling and not (self.pairwise and sub.fixed)
            if resample_here:
                sub.v = self.ico_v.copy()
                sub.faces = self.ico_f
                sub.tree = SphereAABB(sub.v, sub.faces)
            else:
                sub.v = v0.copy()
                sub.faces = faces0
                sub.tree = tree0
            sub.sphere0_v, sub.sphere0_faces = v0, faces0
            nv0 = len(v0)
            nv = len(sub.v)
            raw_prop = np.empty((self.nprop, nv0), np.float32)
            for k in range(self.nprop):
                raw_prop[k] = _read_floats(properties[s][k], nv0)
            if self.resampling:
                fid, bary = tree0.closest_face(sub.v)
                prop = np.empty((self.nprop, nv), np.float32)
                for k in range(self.nprop):
                    K.interpolate(raw_prop[k], faces0, fid, bary, prop[k])
            else:
                prop = raw_prop
            for k in range(self.nprop):
                K.normalize_property(prop[k])
            sub.prop = np.ascontiguousarray(prop)
            # spherical harmonics information
            sub.coeff = self.coeff[s]
            sub.pole = np.array([0.0, math.pi / 2], np.float32)
            sub.pole[1] = np.float32(math.pi / 2)
            if coefficient is not None and not sub.fixed:
                d = read_sh_coefficients(coefficient[s], self.degree, dtype=np.float64)
                n = (d.degree + 1) ** 2
                sub.pole = d.pole.astype(np.float32)
                sub.coeff[:, :n] = d.coeff
            if ico_Y is None or sub.fixed:
                sub.Y = basis(sub.v.astype(np.float64), self.degree)
            else:
                sub.Y = ico_Y
            sub.p = sub.v.copy()
            sub.p0 = np.zeros_like(sub.v)
            self._set_tangent(sub)
            sub.patch_cache = None if self.resampling else _patch_cache(sub.faces)
            sub.flip = np.zeros(len(sub.faces), bool)
            self._init_flips(sub)
            sub.area0 = np.empty(len(sub.faces), np.float32)
            K.face_areas(sub.v, sub.faces, sub.area0)
            sub.area1 = sub.area0.copy()
            sub.tree_cache = np.full(self.nsamples, -1, np.int32)
            sub.step = 1.0
            sub.is_flip = False
            sub.step_adjusted = False
            self.subj.append(sub)
            self._say(f"Subject {s} - {spheres[s]}")
        if self.resampling:
            pivot = None
            for sub in self.subj:
                if not (self.pairwise and sub.fixed):
                    if pivot is None:
                        pivot = _patch_cache(sub.faces)
                    sub.patch_cache = pivot
                else:
                    sub.patch_cache = _patch_cache(sub.faces)
        self.feature = np.zeros((nsubj, self.nprop, self.nsamples), np.float32)
        for s in range(nsubj):
            self.update_deformation(s, True)
        if coefficient is not None:
            for sub in self.subj:
                K.face_areas(sub.v, sub.faces, sub.area0)
                sub.area1 = sub.area0.copy()
        self.mean = np.zeros((self.nprop, self.nsamples))
        self.variance = np.zeros((self.nprop, self.nsamples))
        guess = self.guess_active
        self.guess_active = False
        self.updated[:] = False
        self.update_displacement()
        self.update_properties()
        if self.pairwise:
            self._init_pairwise(tmp_variance)
        self.guess_active = guess
        self.coeff_prev[:] = self.coeff
        self._say("Initialization done!")

    def _set_tangent(self, sub):
        sub.axis0 = K.pole_axis(sub.pole[0], sub.pole[1])
        sub.tan1, sub.tan2 = K.tangent_plane(sub.pole[0], sub.pole[1])

    def _init_flips(self, sub):
        neg, ninit, nflip = K.init_flips(sub.v, sub.faces, sub.flip)
        sub.neg_thr = np.float32(neg + 1e-5)
        return neg, ninit, nflip

    def _ring(self, sub):
        key = id(sub.faces)
        if key not in self._ring_cache:
            nl = vertex_neighbor_lists(sub.faces, len(sub.v))
            self._ring_cache[key] = (nl.indptr, nl.indices)
        return self._ring_cache[key]

    # -------------------------------------------------------------- deformation
    def update_deformation(self, s, enforce=False):
        sub = self.subj[s]
        updated = bool(self.updated[s]) and not enforce
        n = (self.degree_inc + 1) ** 2
        if updated and np.any(self.coeff[s, :, :n] != self.coeff_prev[s, :, :n]):
            updated = False
        if not updated:
            out = K.update_coordinates(sub.p, sub.Y, sub.coeff, self.K, sub.axis0, sub.tan1, sub.tan2)
            unit_rows_inplace(out)
            sub.v = out
        self.updated[s] = updated
        if self.icosahedron < self.fine_res and not sub.fixed:
            K.face_areas(sub.v, sub.faces, sub.area1)

    def update_displacement(self, degree=0):
        for sub in self.subj:
            if sub.fixed:
                continue
            sub.p0 = K.update_coordinates(
                sub.p, sub.Y, sub.coeff, (degree + 1) ** 2, sub.axis0, sub.tan1, sub.tan2
            )

    # ----------------------------------------------------------------- features
    def update_properties(self):
        ns = self.nsamples
        for s, sub in enumerate(self.subj):
            if self.updated[s]:
                continue
            self.updated[s] = True
            sub.tree.update(sub.v)
            fid = np.empty(ns, np.int32)
            bary = np.zeros((ns, 3), np.float32)
            K.patch_search(
                sub.v, sub.faces, sub.patch_cache, sub.tree_cache, self.samples, fid, bary,
                K._nchunk(ns),
            )
            miss = np.flatnonzero(fid == -1)
            if miss.size:
                f2, b2 = sub.tree.closest_face(self.samples[miss])
                if (f2 < 0).any():
                    raise RuntimeError(
                        f"Fatal error: Property - no closest triangle found in subject {s}"
                    )
                fid[miss] = f2
                bary[miss] = b2
            for k in range(self.nprop):
                K.interpolate(sub.prop[k], sub.faces, fid, bary, self.feature[s, k])
            sub.tree_cache = fid
        if not self.guess_active and (self.degree_inc == 0 or self.pairwise):
            self.update_property_stats()

    def update_property_stats(self):
        select = np.array(
            [(self.ndef == self.nsubj) or (self.ndef < self.nsubj and f) for f in self.fixed]
        )
        K.feature_mean(self.feature, select, self.ndef, self.nsubj, self.mean)
        if self.pairwise:
            return
        if self.degree_inc == 0 or self.icosahedron < self.fine_res:
            self.variance[:] = 1.0
            return
        tmp = np.empty(self.nsamples)
        for k in range(self.nprop):
            K.feature_variance(self.feature, self.mean, self.nsubj, k, tmp)
            self.variance[k] = self._floor_variance(smooth_data(self.ico_f, tmp, 3))

    def _floor_variance(self, v):
        m = K.seq_sum_div(v, self.nsamples)
        self.cost_var = m
        m *= 0.1
        if m < 1e-5:
            m = 1.0
        return np.where(v < m, m, v)

    def _init_pairwise(self, tmp_variance):
        if tmp_variance is None:
            self.variance[:] = 1.0
            return
        t = 0 if self.fixed[0] else 1
        sub = self.subj[t]
        nv = len(sub.v)
        var = _read_floats(tmp_variance, nv * self.nprop).reshape(self.nprop, nv)
        bary = np.empty((self.nsamples, 3), np.float32)
        K.pairwise_bary(sub.v, sub.faces, sub.tree_cache, self.samples, bary)
        for k in range(self.nprop):
            out = np.empty(self.nsamples, np.float32)
            K.interpolate(var[k], sub.faces, sub.tree_cache, bary, out)
            self.variance[k] = self._floor_variance(smooth_data(self.ico_f, out.astype(np.float64), 3))

    # ------------------------------------------------------------------ energy
    def trace(self):
        self.update_properties()
        active = ~self.fixed
        self.cost_prop = K.var_properties(
            self.feature, self.mean, self.variance, active, self.nprop,
            self.pairwise and self.degree_inc == 0,
        )
        fine = self.icosahedron >= self.fine_res
        if self.degree_inc == 0 or fine or self.lambda2 == 0:
            self.cost_area = 0.0
        else:
            c = 0.0
            for sub in self.subj:
                if not sub.fixed:
                    c = K.var_area_add(sub.area0, sub.area1, c)
            self.cost_area = c / self.nsubj
        if self.degree_inc == 0 or not fine or self.lambda2 == 0:
            self.cost_disp = 0.0
        else:
            c = 0.0
            for sub in self.subj:
                if not sub.fixed:
                    c = K.var_displacement_add(sub.p0, sub.v, c)
            self.cost_disp = c / self.nsubj
        cost_lm = 0.0  # uninitialised upstream when there are no landmarks
        if not fine:
            return cost_lm + self.eta * self.cost_prop + self.lambda1 * self.cost_area
        return cost_lm + self.eta * self.cost_prop + self.lambda2 * self.cost_disp

    def cost(self):
        tol = 1e-5
        for i, sub in enumerate(self.subj):
            sub.is_flip = False
            sub.step_adjusted = False
            if sub.fixed or sub.step == 0:
                continue
            self.update_deformation(i)
            sub.is_flip = (
                False
                if self.degree_inc == 0
                else bool(K.test_triangle_flip(sub.v, sub.faces, sub.flip, sub.neg_thr))
            )
            while self.degree_inc != 0 and sub.is_flip and sub.step > tol:
                self.coeff[i][:] = self.coeff_prev[i]
                if not self.pairwise:
                    if sub.is_flip:
                        sub.step *= 0.5
                    if sub.step <= tol:
                        sub.step = 0
                    else:
                        sub.step_adjusted = True
                        n = (self.degree_inc + 1) ** 2
                        self.coeff[i][:, :n] -= sub.step * self.gradient[i][:, :n]
                self.update_deformation(i, True)
                if self.pairwise:
                    break
                sub.is_flip = bool(K.test_triangle_flip(sub.v, sub.faces, sub.flip, sub.neg_thr))
        if all(sub.is_flip for sub in self.subj if not sub.fixed):
            return float(self.mincost)
        return self.trace()

    # ---------------------------------------------------------------- gradient
    def update_gradient(self, deg_beg, deg_end, lam):
        n, n0 = (deg_end + 1) ** 2, deg_beg**2
        size = n - n0
        fine = self.icosahedron >= self.fine_res
        self.gradient[:] = 0
        self.hessian = [None] * self.nsubj
        unit_var = self.pairwise and self.degree_inc == 0
        norm_p = self.eta * 1.0 / float((self.nprop * self.nsamples) * self.nsubj)
        norm_d = self.lambda2 / float(self.nsubj)
        for s, sub in enumerate(self.subj):
            if sub.fixed or sub.step == 0:
                continue
            H = np.zeros((3 * size, 3 * size)) if fine else None
            ring = self._ring(sub)
            for k in range(self.nprop):
                g, Hk = G.accumulate_properties(
                    sub, k, deg_beg, deg_end, samples=self.samples, mean_k=self.mean[k],
                    var_k=self.variance[k], feat_k=self.feature[s, k], normalization=norm_p,
                    fine=fine, unit_var=unit_var, axis0=sub.axis0, ring=ring,
                )
                for c in range(3):
                    self.gradient[s, c, n0:n] += g[c * size : (c + 1) * size] * norm_p
                if fine:
                    H += Hk
            if self.degree_inc > 0 and fine:
                g, Hd = G.accumulate_displacement(
                    sub, deg_beg, deg_end, normalization=norm_d, fine=fine, axis0=sub.axis0
                )
                nv = len(sub.v)
                for c in range(3):
                    self.gradient[s, c, n0:n] += g[c * size : (c + 1) * size] * norm_d / nv
                H += Hd
            self.hessian[s] = H
        if fine:
            for s, sub in enumerate(self.subj):
                if sub.fixed or sub.step == 0:
                    continue
                self._update_new_gradient(deg_beg, deg_end, lam, s)

    def _update_new_gradient(self, deg_beg, deg_end, lam, s):
        n, n0 = (deg_end + 1) ** 2, deg_beg**2
        size = n - n0
        alpha = (
            0.001 / math.exp(self.n_success)
            if (self.icosahedron >= self.fine_res and deg_beg == 0 and deg_end == self.degree)
            else 0.0
        )
        Gv = np.concatenate([self.gradient[s, c, n0:n] for c in range(3)])
        x = G.solve_direction(self.hessian[s], Gv, lam, alpha)
        for c in range(3):
            self.gradient[s, c, n0:n] = x[c * size : (c + 1) * size]

    # ------------------------------------------------------------ optimisation
    def min_gradient_descent(self, deg_beg, deg_end):
        self.n_success = 0
        t = 1.0
        self.mincost = np.float32(self.cost())
        self._say(f"[{self.n_iter}] {float(self.mincost):g}")
        self.history.append((self.n_iter, float(self.mincost)))
        self.n_iter += 1
        success = True
        tol = 1e-5
        advance = True
        for sub in self.subj:
            sub.step = 1.0
        nn = (self.degree_inc + 1) ** 2
        while t > tol:
            if (
                self.n_iter > self.max_iter
                and self.icosahedron >= self.fine_res
                and deg_beg == 0
                and deg_end == self.degree
            ):
                break
            if success:
                self.update_gradient(deg_beg, deg_end, 0.001 * t / (self.n_success + 1))
            for i, sub in enumerate(self.subj):
                if sub.step == 0:
                    continue
                self.coeff[i][:, :nn] -= sub.step * self.gradient[i][:, :nn]
            current = self.cost()
            if current < float(self.mincost):
                if self.realtime_coeff and self.output_coeff is not None:
                    for s, sub in enumerate(self.subj):
                        if not sub.fixed:
                            self.save_coeff(self.output_coeff[s], s)
                self.coeff_prev[:] = self.coeff
                if advance:
                    t *= 2
                    for sub in self.subj:
                        sub.step *= 2
                success = True
                diff = float(self.mincost) - current
                last = self.cost_prop, (self.cost_area if self.icosahedron < self.fine_res else self.cost_disp)
                flips = "".join(f" {i}" for i, sb in enumerate(self.subj) if sb.is_flip)
                self._say(f"[{self.n_iter}] {current:g}: {last[0]:g} {last[1]:g}{flips}")
                self.history.append((self.n_iter, current))
                if diff < tol:
                    self.mincost = np.float32(current)
                    break
                self.mincost = np.float32(current)
                self.n_success += 1
            else:
                self.coeff[:] = self.coeff_prev
                for i in range(self.nsubj):
                    self.update_deformation(i, True)
                self.update_properties()
                t *= 0.5
                for sub in self.subj:
                    if not sub.step_adjusted:
                        sub.step *= 0.5
                    if sub.step <= tol:
                        sub.step = 0
                    sub.step_adjusted = False
                success = False
                advance = False
            self.n_iter += 1
        self.coeff[:] = self.coeff_prev

    def optimization(self):
        if self.guess_active:
            self._say("Guess initial coeffs.. ")
            guess_init_coeff(self)
            self._say("done")
        while self.degree_inc <= self.degree:
            self.n_iter = 0
            self._say(f"Phase: {self.degree_inc}")
            if self.icosahedron < self.fine_res:
                self.min_gradient_descent(0, self.degree_inc)
            else:
                self.min_gradient_descent(self.degree_inc, self.degree_inc)
            if self.degree_inc == self.degree:
                break
            self.degree_inc = min(self.degree_inc + 1, self.degree)
            if self.degree_inc == 1:
                for sub in self.subj:
                    self._init_flips(sub)
                self.update_displacement(self.degree)
                self.update_property_stats()
        self._say("Phase: Final")
        self.n_iter = 0
        self.degree_inc = self.degree
        self.update_property_stats()
        self.min_gradient_descent(0, self.degree_inc)

    def run(self):
        self._say("Optimization")
        self.optimization()
        self._say("Saving outputs")
        for s, sub in enumerate(self.subj):
            self._say(f"-Subject {s}")
            if self.output is not None and not sub.fixed:
                self.update_deformation(s, True)
                self.save_sphere(self.output[s], s)
            if self.output_coeff is not None and not sub.fixed:
                self.save_coeff(self.output_coeff[s], s)
        self._say("All done!")

    # ------------------------------------------------------------------ output
    def save_coeff(self, filename, s):
        sub = self.subj[s]
        write_sh_coefficients(
            filename, ShDeformation(self.degree, np.asarray(sub.pole, np.float32), sub.coeff.copy())
        )

    def deformed_sphere(self, s):
        """Vertices written by ``saveSphere`` (original sphere topology)."""
        sub = self.subj[s]
        if not self.resampling:
            return sub.v
        v0 = sub.sphere0_v
        Y = basis(v0.astype(np.float64), self.degree)
        out = K.update_coordinates(v0, Y, sub.coeff, self.K, sub.axis0, sub.tan1, sub.tan2)
        unit_rows_inplace(out)
        return out

    def save_sphere(self, filename, s):
        sub = self.subj[s]
        _write_vtk_meshlib(filename, self.deformed_sphere(s), sub.sphere0_faces)
