// Oracle harness for MeshLib geodesics (B2.6).  Links MeshLib dcef807 (-ffp-contract=off build).
//
//   harness_geodesic mesh.vtk queries.txt out.txt
//
// queries.txt, one per line:
//   D <start> <dmax>        Geodesic::perform_front_propagation(start, double dmax); prints dist[] (all vertices)
//   E <start> <end>         perform_front_propagation(start, int end);               prints dist[]
//   M <start> <nstart2>...  not used
//   P <start> <end>         TRACE's saveGeodesicPath step: propagate(start, end), GeodesicPath::computeGeodesicPath(end);
//                           prints the path (xyz and barycentric rows)
// Output (text): "<tag> <args...> n" then n lines (D/E: "%.17g" per vertex; P: "x y z | a b c").
// State/source arrays are also printed for D/E ("S" and "Q" lines) because TRACE's distance matrix depends on
// which vertices were finalised before the stop.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "Mesh.h"
#include "Geodesic/Geodesic.h"
#include "Geodesic/GeodesicPath.h"

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: %s mesh.vtk queries.txt out.txt\n", argv[0]); return 2; }
	Mesh mesh;
	mesh.openFile(argv[1]);
	Geodesic geo(&mesh);
	GeodesicPath gp(geo.dist(), &mesh);
	FILE *fq = fopen(argv[2], "r");
	FILE *fo = fopen(argv[3], "w");
	char tag[8];
	const int n = mesh.nVertex();
	while (fscanf(fq, "%7s", tag) == 1)
	{
		if (tag[0] == 'D')
		{
			int s; double dmax;
			if (fscanf(fq, "%d %lf", &s, &dmax) != 2) break;
			geo.perform_front_propagation(s, dmax);
			fprintf(fo, "D %d %.17g %d\n", s, dmax, n);
			for (int i = 0; i < n; i++) fprintf(fo, "%.17g %d %d\n", geo.dist()[i], geo.state()[i], geo.source()[i]);
		}
		else if (tag[0] == 'E')
		{
			int s, e;
			if (fscanf(fq, "%d %d", &s, &e) != 2) break;
			geo.perform_front_propagation(s, e);
			fprintf(fo, "E %d %d %d\n", s, e, n);
			for (int i = 0; i < n; i++) fprintf(fo, "%.17g %d %d\n", geo.dist()[i], geo.state()[i], geo.source()[i]);
		}
		else if (tag[0] == 'P')
		{
			int s, e;
			if (fscanf(fq, "%d %d", &s, &e) != 2) break;
			geo.perform_front_propagation(s, e);
			gp.computeGeodesicPath(e);
			int m = gp.size();
			fprintf(fo, "P %d %d %d\n", s, e, m);
			for (int i = 0; i < m; i++)
			{
				const float *p = gp.getPoint(i);
				const float *b = gp.getBarycentricPoint(i);
				fprintf(fo, "%.9g %.9g %.9g | %.9g %.9g %.9g\n", p[0], p[1], p[2], b[0], b[1], b[2]);
			}
		}
	}
	fclose(fq); fclose(fo);
	return 0;
}
