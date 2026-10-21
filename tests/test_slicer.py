# ruff: noqa: E501
"""Tests for pybiscuit.mesh.slicer (MeshLib AABB tree + Slicer port) against the harness golden."""

from __future__ import annotations

import json

import numpy as np

from conftest import REPO
from fixtures.meshes import folded_sphere
from pybiscuit.mesh.slicer import Slicer

GOLD = REPO / "tests" / "data" / "golden" / "slicer" / "fold_l4"


def parse(path):
    lines = path.read_text().split("\n")
    i, out = 0, []
    while i < len(lines) and lines[i]:
        ng = int(lines[i].split()[1])
        i += 1
        gs = []
        for _ in range(ng):
            n = int(lines[i].split()[1])
            rows = [[float(x) for x in r.replace("|", " ").split()] for r in lines[i + 1 : i + 1 + n]]
            gs.append(np.array(rows).reshape(n, 6))
            i += 1 + n
        out.append(gs)
    return out


def test_planes_match_meshlib_slicer():
    f = json.loads((GOLD / "command.json").read_text())["fixture"]
    g = folded_sphere(f["a"], f["l"], f["m"], f["level"])
    v = (g.vertices * f["radius"]).astype(np.float32)
    planes = np.loadtxt(GOLD / "planes.txt", dtype=np.float32)
    ref = parse(GOLD / "out.txt")
    sl = Slicer(v, g.faces)
    n_pts = 0
    reordered = 0
    for pl, groups in zip(planes, ref, strict=True):
        got = [(p, e) for p, e in sl.slicing(*pl)]
        key = lambda size, first: (size, tuple(float(x) for x in np.round(np.asarray(first, dtype=np.float64), 3)))  # noqa: E731
        order_ref = [key(r.shape[0], r[0, :3]) for r in groups]
        order_got = [key(len(p), p[0]) for p, _ in got]
        assert sorted(order_got) == sorted(order_ref)  # same contours
        if order_got != order_ref:  # equal start-edge ids: std::sort tie order (libc++-specific)
            reordered += 1
            idx = [order_got.index(k) for k in order_ref]
            got = [got[i] for i in idx]
        for (p, e), r in zip(got, groups, strict=True):
            assert np.abs(p - r[:, :3]).max(initial=0) <= 1e-5  # %.9g print of float32
            assert np.array_equal(e, r[:, 3:].astype(np.float32))
            n_pts += len(p)
    assert n_pts > 1000
    assert reordered <= 2


def test_tree_leaves_cover_all_faces():
    from pybiscuit.mesh.slicer import build_tree

    g = folded_sphere(0.1, 4, 2, 3)
    left, right, tbox, ss, sl, pool = build_tree(g.vertices * 50, g.faces)
    leaves = np.flatnonzero(left == -1)
    covered = np.unique(np.concatenate([pool[ss[k] : ss[k] + sl[k]] for k in leaves]))
    assert covered.size == g.faces.shape[0]
