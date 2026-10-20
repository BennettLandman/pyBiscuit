// Oracle harness for MeshLib SphericalHarmonics::basis (dcef807).
// usage: harness_sh <points_f32.bin (N*3 float32, LE)> <N> <degree> <from> <out_f32.bin> <out_f64.bin>
// out_f32: float overload   -> N * ((degree+1)^2 - from^2) float32
// out_f64: double overload (points widened float->double, as HSD does) -> same count, float64
// Build (see scripts/oracles/build_sh.sh):
//   clang++ -O2 -std=c++14 -include cstddef -I$ML/include/Util harness_sh.cpp $ML/src/Util/Geom.cpp \
//           $ML/src/Util/SphericalHarmonics.cpp -o harness_sh
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "SphericalHarmonics.h"

int main(int argc, char **argv)
{
	if (argc != 7) { fprintf(stderr, "usage: %s pts.bin N degree from out32 out64\n", argv[0]); return 2; }
	int N = atoi(argv[2]), degree = atoi(argv[3]), from = atoi(argv[4]);
	int K = (degree + 1) * (degree + 1) - from * from;
	std::vector<float> p((size_t)N * 3);
	FILE *fp = fopen(argv[1], "rb");
	if (!fp || fread(p.data(), sizeof(float), p.size(), fp) != p.size()) { fprintf(stderr, "read error\n"); return 1; }
	fclose(fp);
	std::vector<float> y32((size_t)N * K);
	std::vector<double> y64((size_t)N * K);
	for (int i = 0; i < N; i++)
	{
		SphericalHarmonics::basis(degree, &p[(size_t)i * 3], &y32[(size_t)i * K], from);
		double d[3] = {p[(size_t)i * 3], p[(size_t)i * 3 + 1], p[(size_t)i * 3 + 2]};
		SphericalHarmonics::basis(degree, d, &y64[(size_t)i * K], from);
	}
	fp = fopen(argv[5], "wb"); fwrite(y32.data(), sizeof(float), y32.size(), fp); fclose(fp);
	fp = fopen(argv[6], "wb"); fwrite(y64.data(), sizeof(double), y64.size(), fp); fclose(fp);
	return 0;
}
