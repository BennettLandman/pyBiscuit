// Oracle harness for the gw triangular interpolations (row B2.7).  Links MeshLib dcef807
// (-ffp-contract=off build, see build_curve_extraction_nofma.sh).
//
//   harness_interpolation mesh.vtk dist.txt queries.txt out.txt
//
// The GW mesh is built exactly as Geodesic::setupMesh does (float VTK vertices widened to double,
// faces in file order, GW_Mesh::BuildConnectivity).  dist.txt holds one "%.17g" distance per vertex,
// assigned with GW_GeodesicVertex::SetDistance (the interpolation only reads distances, so they need
// not come from a fast marching run).
//
// queries.txt, one per line:
//   <kind> <face> <perm> <x> <y>
//     kind  L / Q / C  -> GW_TriangularInterpolation_{Linear,Quadratic,Cubic}
//     perm  0..5       -> which face vertices are passed as (v0, v1, v2) to ComputeValue/Gradient
//                         (PERM below), as GW_GeodesicPath passes them in path order
// A fresh interpolation object is set up on the face for every query.
// Output, one line per query: "<value> <dx> <dy> <c0> ... <c5>" (%.17g; coefficients 0 for L).
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <vector>
#include "Mesh.h"
using std::string;  // as Geodesic.h, which the GW headers rely on
#define private public  // read the fitted coefficients
#include "gw/config.h"
#include "gw/gw_core/GW_Config.h"
#include "gw/gw_core/GW_MathsWrapper.h"
#include "gw/gw_geodesic/GW_GeodesicMesh.h"
#include "gw/gw_geodesic/GW_TriangularInterpolation_Linear.h"
#include "gw/gw_geodesic/GW_TriangularInterpolation_Quadratic.h"
#include "gw/gw_geodesic/GW_TriangularInterpolation_Cubic.h"
#undef private

using namespace GW;

static const int PERM[6][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}, {0, 2, 1}, {2, 1, 0}, {1, 0, 2}};

int main(int argc, char **argv)
{
	if (argc < 5) { fprintf(stderr, "usage: %s mesh.vtk dist.txt queries.txt out.txt\n", argv[0]); return 2; }
	Mesh mesh;
	mesh.openFile(argv[1]);
	const int nv = mesh.nVertex(), nf = mesh.nFace();
	GW_GeodesicMesh gw;
	gw.SetNbrVertex(nv);
	for (int i = 0; i < nv; ++i)
	{
		GW_GeodesicVertex &v = (GW_GeodesicVertex &) gw.CreateNewVertex();
		v.SetPosition(GW_Vector3D((*mesh.vertex(i))[0], (*mesh.vertex(i))[1], (*mesh.vertex(i))[2]));
		gw.SetVertex(i, &v);
	}
	gw.SetNbrFace(nf);
	for (int i = 0; i < nf; ++i)
	{
		GW_GeodesicFace &f = (GW_GeodesicFace &) gw.CreateNewFace();
		f.SetVertex(*gw.GetVertex((*mesh.face(i))[0]), *gw.GetVertex((*mesh.face(i))[1]), *gw.GetVertex((*mesh.face(i))[2]));
		gw.SetFace(i, &f);
	}
	gw.BuildConnectivity();

	FILE *fd = fopen(argv[2], "r");
	for (int i = 0; i < nv; ++i)
	{
		double d;
		if (fscanf(fd, "%lf", &d) != 1) { fprintf(stderr, "short dist file\n"); return 1; }
		((GW_GeodesicVertex *) gw.GetVertex(i))->SetDistance(d);
	}
	fclose(fd);

	FILE *fq = fopen(argv[3], "r");
	FILE *fo = fopen(argv[4], "w");
	char kind[4];
	int face, perm;
	double x, y;
	while (fscanf(fq, "%3s %d %d %lf %lf", kind, &face, &perm, &x, &y) == 5)
	{
		GW_GeodesicFace &F = *(GW_GeodesicFace *) gw.GetFace(face);
		GW_GeodesicVertex *v[3];
		for (int k = 0; k < 3; ++k) v[k] = (GW_GeodesicVertex *) F.GetVertex(PERM[perm][k]);
		double val, dx, dy, c[6] = {0, 0, 0, 0, 0, 0};
		if (kind[0] == 'L')
		{
			GW_TriangularInterpolation_Linear I;
			I.SetUpTriangularInterpolation(F);
			val = I.ComputeValue(*v[0], *v[1], *v[2], x, y);
			I.ComputeGradient(*v[0], *v[1], *v[2], x, y, dx, dy);
		}
		else if (kind[0] == 'Q')
		{
			GW_TriangularInterpolation_Quadratic I;
			I.SetUpTriangularInterpolation(F);
			val = I.ComputeValue(*v[0], *v[1], *v[2], x, y);
			I.ComputeGradient(*v[0], *v[1], *v[2], x, y, dx, dy);
			for (int k = 0; k < 6; ++k) c[k] = I.Coeffs[k];
		}
		else
		{
			GW_TriangularInterpolation_Cubic I;
			I.SetUpTriangularInterpolation(F);
			val = I.ComputeValue(*v[0], *v[1], *v[2], x, y);
			I.ComputeGradient(*v[0], *v[1], *v[2], x, y, dx, dy);
			for (int k = 0; k < 6; ++k) c[k] = I.Coeffs[k];
		}
		fprintf(fo, "%.17g %.17g %.17g", val, dx, dy);
		for (int k = 0; k < 6; ++k) fprintf(fo, " %.17g", c[k]);
		fprintf(fo, "\n");
	}
	fclose(fq);
	fclose(fo);
	return 0;
}
