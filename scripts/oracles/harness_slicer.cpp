// Oracle harness for MeshLib Slicer (plane/mesh contours; used by SulcalPoint/GyralPoint).  -ffp-contract=off build.
//
//   harness_slicer mesh.vtk planes.txt out.txt
//
// planes.txt: one plane per line "a b c d" (plane a*x+b*y+c*z+d = 0, floats).
// out.txt: per plane "S <ngroups>"; per group "G <size>" followed by size lines "x y z | id0 id1 id2" (%.9g).
#include <cstdio>
#include "Mesh.h"
#include "Slicer.h"

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: %s mesh.vtk planes.txt out.txt\n", argv[0]); return 2; }
	Mesh mesh;
	mesh.openFile(argv[1]);
	Slicer slicer(&mesh);
	FILE *fq = fopen(argv[2], "r");
	FILE *fo = fopen(argv[3], "w");
	float a, b, c, d;
	while (fscanf(fq, "%f %f %f %f", &a, &b, &c, &d) == 4)
	{
		int ng = slicer.slicing(a, b, c, d);
		fprintf(fo, "S %d\n", ng);
		for (int g = 0; g < ng; g++)
		{
			int n = slicer.size(g);
			float *slice = new float[n * 3 + 3];
			slicer.getSlice(slice, g);
			fprintf(fo, "G %d\n", n);
			for (int i = 0; i < n; i++)
			{
				const float *id = slicer.getEdgeID(i, g);
				fprintf(fo, "%.9g %.9g %.9g | %.9g %.9g %.9g\n", slice[i * 3], slice[i * 3 + 1], slice[i * 3 + 2], id[0], id[1], id[2]);
			}
			delete[] slice;
		}
	}
	fclose(fq); fclose(fo);
	return 0;
}
