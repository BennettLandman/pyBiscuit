"""Hemisphere separation (B5.2) on synthetic bridged ellipsoids (oracle-free)."""

import numpy as np
import pytest

from fixtures.bridged import bridged_ellipsoids
from pybiscuit.mesh.topology import topology_summary
from pybiscuit.surfprep import contour
from pybiscuit.surfprep.hemisphere import (
    HemisphereError,
    disk_patch,
    separate_hemispheres,
)


@pytest.fixture(scope="module")
def split():
    mesh = bridged_ellipsoids(level=4)
    return mesh, separate_hemispheres(mesh)


def test_both_hemispheres_closed_genus0_oriented(split):
    _, r = split
    for h in (r.lh, r.rh):
        t = topology_summary(h)
        assert t.is_closed and t.is_manifold and t.is_consistently_oriented
        assert t.n_components == 1 and t.genus == 0


def test_left_is_smaller_x(split):
    _, r = split
    assert r.lh.vertices[:, 0].mean() < 0 < r.rh.vertices[:, 0].mean()


def test_mask_zero_exactly_on_patch(split):
    mesh, r = split
    d = r.diagnostics
    n_patch = sum(d.n_patch.values()) - sum(d.loop_sizes)  # interior patch vertices
    assert (r.lh_mask == 0).sum() + (r.rh_mask == 0).sum() == n_patch
    assert r.lh_mask.size == r.lh.n_vertices and r.rh_mask.size == r.rh.n_vertices
    assert set(np.unique(r.lh_mask)) <= {0, 1}
    # every original vertex in the map sits at mask 1
    for h, mask in ((0, r.lh_mask), (1, r.rh_mask)):
        ids = r.hemi_map[r.hemi_map[:, 2] == h, 1]
        assert mask[ids].all()
        assert ids.size == (mask == 1).sum()


def test_hemi_map_round_trips_original_indices(split):
    mesh, r = split
    assert r.hemi_map.shape[1] == 3
    orig = r.hemi_map[:, 0]
    assert np.all(np.diff(orig) > 0)  # whole-brain order
    assert orig.min() >= 0 and orig.max() < mesh.n_vertices
    same = 0
    for h, m in ((0, r.lh), (1, r.rh)):
        sel = r.hemi_map[:, 2] == h
        pos = m.vertices[r.hemi_map[sel, 1]]
        same += (np.abs(pos - mesh.vertices[orig[sel]]).max(axis=1) < 1e-9).sum()
    assert same > 0.8 * orig.size  # only the seam neighbourhood moves


def test_plane_is_in_the_neck(split):
    _, r = split
    assert abs(r.diagnostics.plane_offset) < 0.6
    assert r.diagnostics.plane_rank == 0


def test_rule_alternatives_are_selectable():
    mesh = bridged_ellipsoids(level=4)
    r = separate_hemispheres(mesh, rule="fewest_faces")
    assert r.diagnostics.plane_rule == "fewest_faces"
    with pytest.raises(ValueError):
        separate_hemispheres(mesh, rule="nope")


def test_candidate_offsets_midpoints():
    x = np.linspace(-10, 10, 2001)
    off = contour.candidate_offsets(x)
    assert off.size == 101
    assert np.all(np.abs(-off - x.mean()) <= 0.5 + 0.01)


def test_mesh_contour_ranks_by_length():
    mesh = bridged_ellipsoids(level=4)
    offs = np.array([-35.0, 0.0, 20.0])
    ranking, cut = contour.mesh_contour(mesh.vertices, mesh.faces, [1, 0, 0], offs)
    assert ranking[0] == 1  # x = 0 is the neck
    assert all(len(c) > 0 for c in cut)


def test_disk_patch_is_a_disk():
    pts, tri, bid = disk_patch(40)
    from pybiscuit.mesh.topology import topology_summary as ts

    t = ts(np.zeros((pts.shape[0], 3)), tri)
    assert t.n_boundary_loops == 1 and t.genus == 0 and t.n_components == 1
    assert bid.size == 40


def test_no_neck_fails_cleanly():
    mesh = bridged_ellipsoids(level=3)
    mesh.vertices[:, 0] *= 0.0  # degenerate: every candidate plane cuts everything
    with pytest.raises((HemisphereError, ValueError, np.linalg.LinAlgError)):
        separate_hemispheres(mesh)
