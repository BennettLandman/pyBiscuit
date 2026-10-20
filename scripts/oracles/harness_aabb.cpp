// Oracle harness for pybiscuit.sphere.aabb (brief 02, row B2.5).
//
// Links against upstream MeshLib (ilwoolyu/MeshLib) unmodified.  The only
// build-time trick is `#define private public` around the AABB_Sphere include
// so that the private closestFace overloads can be called; no numeric code is
// touched.  Build with scripts/oracles/build_harness_aabb.sh (uses
// -ffp-contract=off so the arm64 build does not fuse multiply-adds, matching
// the x86-64 reference image).
//
// usage:
//   harness_aabb MODE MESH.vtk QUERIES.txt OUT.txt [DEFORMED.vtk [refresh]]
//
//   MODE   closest            AABB_Sphere::closestFace(v, coeff)   (public)
//          bool:E1:E2         closestFaceBool(v, coeff, fresh visited, E1, E2)
//          xcand:E1:E2        closestFace(v, coeff, empty xcand, E1, E2)
//          search:E1:E2       searchTree(p, &cand, E1, E2)  (candidate list)
//          leaves             dump the tree: one line per leaf "n f0 f1 ..." followed by
//                             its box "phi0 phi1 theta0 theta1" (QUERIES.txt is ignored)
//   QUERIES.txt   one query per line "x y z" (float32 values, %.9g)
//   OUT.txt       closest/bool/xcand: "idx c0 c1 c2" (%.9g; idx -1 => coeff
//                 printed as 0 0 0).   search: "n i0 i1 ... in-1" (sorted unique)
//   DEFORMED.vtk  optional: after building the tree on MESH.vtk, copy the
//                 vertices of DEFORMED.vtk into the mesh with Vertex::setVertex
//                 (as SphericalRemeshing.cpp does) and call AABB_Sphere::update().
//                 If the 6th argument is the word "refresh", Mesh::updateNormal()
//                 is called before update() (SphericalRemesh/HSD do NOT do this).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <iomanip>
#include <set>
#include <iterator>

#define private public
#define protected public
#include "Mesh.h"
#include "AABB_Sphere.h"
#undef private
#undef protected

using namespace std;

int main(int argc, char **argv)
{
	if (argc < 5)
	{
		fprintf(stderr, "usage: %s MODE MESH.vtk QUERIES.txt OUT.txt [DEFORMED.vtk [refresh]]\n", argv[0]);
		return 2;
	}
	string mode = argv[1];
	float e1 = 0, e2 = 0;
	string kind = mode;
	size_t p1 = mode.find(':');
	if (p1 != string::npos)
	{
		kind = mode.substr(0, p1);
		size_t p2 = mode.find(':', p1 + 1);
		e1 = strtof(mode.substr(p1 + 1, p2 - p1 - 1).c_str(), NULL);
		e2 = (p2 == string::npos) ? e1 : strtof(mode.substr(p2 + 1).c_str(), NULL);
	}

	Mesh mesh;
	mesh.openFile(argv[2]);
	if (mesh.nFace() == 0) { fprintf(stderr, "failed to read mesh\n"); return 1; }

	AABB_Sphere tree(&mesh);

	if (argc >= 6)
	{
		Mesh def;
		def.openFile(argv[5]);
		if (def.nVertex() != mesh.nVertex()) { fprintf(stderr, "deformed mesh size mismatch\n"); return 1; }
		for (int i = 0; i < mesh.nVertex(); i++)
			((Vertex *)mesh.vertex(i))->setVertex(def.vertex(i)->fv());
		if (argc >= 7 && !strcmp(argv[6], "refresh")) mesh.updateNormal();
		tree.update();
	}

	if (kind == "leaves")
	{
		FILE *fl = fopen(argv[4], "w");
		vector<AABB_Sphere::node *> st;
		st.push_back(tree.m_tree);
		while (!st.empty())
		{
			AABB_Sphere::node *nd = st.back(); st.pop_back();
			if (nd->left == NULL && nd->right == NULL)
			{
				vector<int> c = nd->cand;
				sort(c.begin(), c.end());
				fprintf(fl, "%d", (int)c.size());
				for (size_t i = 0; i < c.size(); i++) fprintf(fl, " %d", c[i]);
				fprintf(fl, " %.9g %.9g %.9g %.9g\n", nd->phi0, nd->phi1, nd->theta0, nd->theta1);
			}
			else
			{
				if (nd->left) st.push_back(nd->left);
				if (nd->right) st.push_back(nd->right);
			}
		}
		fprintf(fl, "maxPhi %.9g pole0 %d pole1 %d minTheta %.9g maxTheta %.9g\n", tree.m_maxPhi, tree.m_pole0, tree.m_pole1, tree.m_minTheta, tree.m_maxTheta);
		fclose(fl);
		return 0;
	}

	FILE *fq = fopen(argv[3], "r");
	if (!fq) { fprintf(stderr, "cannot open queries\n"); return 1; }
	FILE *fo = fopen(argv[4], "w");
	if (!fo) { fprintf(stderr, "cannot open output\n"); return 1; }

	float v[3];
	while (fscanf(fq, "%f %f %f", &v[0], &v[1], &v[2]) == 3)
	{
		float coeff[3] = {0, 0, 0};
		if (kind == "search")
		{
			vector<int> cand;
			tree.searchTree(v, &cand, e1, e2);
			sort(cand.begin(), cand.end());
			cand.erase(unique(cand.begin(), cand.end()), cand.end());
			fprintf(fo, "%d", (int)cand.size());
			for (size_t i = 0; i < cand.size(); i++) fprintf(fo, " %d", cand[i]);
			fprintf(fo, "\n");
			continue;
		}
		int idx;
		if (kind == "closest")
			idx = tree.closestFace(v, coeff);
		else if (kind == "bool")
		{
			vector<char> seen(mesh.nFace() ? mesh.nFace() : 1);
			bool *visited = new bool[mesh.nFace()];
			memset(visited, 0, mesh.nFace() * sizeof(bool));
			idx = tree.closestFaceBool(v, coeff, visited, e1, e2);
			delete [] visited;
		}
		else if (kind == "xcand")
		{
			vector<int> xcand;
			idx = tree.closestFace(v, coeff, xcand, e1, e2);
		}
		else { fprintf(stderr, "unknown mode %s\n", kind.c_str()); return 2; }
		if (idx == -1) coeff[0] = coeff[1] = coeff[2] = 0;
		fprintf(fo, "%d %.9g %.9g %.9g\n", idx, coeff[0], coeff[1], coeff[2]);
	}
	fclose(fq);
	fclose(fo);
	return 0;
}
