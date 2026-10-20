# B2.3 topology (fragment)

Module: `src/pybiscuit/mesh/topology.py`. Tests: `tests/test_topology.py` (27 tests).

## Tests
- Synthetic (inline generators: icosphere, UV sphere, torus, cut spheres, flipped faces, two spheres, Moebius strip, bowtie, non-manifold fan): chi=2/genus 0 on spheres; chi=0/genus 1 on torus; 1 boundary loop on a sphere with one cap removed, 2 loops with both caps removed (chi=0); a single flipped face is the only face reported by `check_orientation` (3 inconsistent edges), flipping it back restores consistency; Moebius strip reported non-orientable; 2 components on two disjoint spheres (chi=4); unreferenced vertices ignored in chi/components; 327,680-face performance test.
- `@pytest.mark.reference` (10 cases): `sphere_{5120,20480,81920,327680}.vtk` and the six `reg/{hcp1058,mb101,oasis45}/{lh,rh}.white.vtk` (note: the files live under `.../opt/biscuit/sphere/`, not `biscuit/biscuit/sphere/`). All: chi=2, genus 0, one component, closed manifold, consistently oriented.

## Timings (327,680 faces / 163,842 vertices, this Mac, pure NumPy/SciPy)
unique_edges 0.11 s, one_ring 0.08 s, vertex_faces 0.04 s, vertex_neighbors 0.13 s, face_adjacency 0.11 s, boundary_loops 0.10 s, connected_components 0.02 s, check_orientation 0.18 s, topology_summary (everything) 0.28 s. Numba not needed.

## Findings / surprises
- All three template white surfaces (hcp1058, mb101, oasis45; both hemispheres) are genus 0 closed manifolds; no exceptions. All shipped spheres likewise.
- All reference meshes have consistent winding.

## Design notes
- Orientation: `is_consistent` = no directed edge used twice. `flipped_faces` is the minority winding class per face-connected component (BFS parity over manifold edges, tie keeps lowest-numbered face's class); a flipped face adjacent to it is not blamed.
- `one_ring` is a vectorised fan walk using face winding; vertices with pinches, non-manifold edges, or inconsistent winding get `valid=False` and an empty ring.
- Unreferenced vertices are excluded from chi and get component label -1.
- Genus is per-component `(2 - chi_c - b_c)/2`, summed; meaningful only when `is_manifold`.
- Boundary loops at pinch vertices are split deterministically but arbitrarily.
