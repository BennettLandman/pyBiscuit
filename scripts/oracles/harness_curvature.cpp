// harness_curvature.cpp -- oracle driver for pybiscuit.mesh.curvature / sphere_map (row B2.4b).
//
// Dumps MeshLib (Mesh.cpp, Util/SurfaceUtil.cpp, Util/Geom.cpp) intermediates and results
// for one .vtk mesh as raw little-endian arrays: <outdir>/<name>.{f32,f64,i32}.
//
// usage: harness_curvature in.vtk outdir [tensor_iters=0,3] [sphere_iters=1,2,5,20,100,10000]
//
// build (macOS arm64; -ffp-contract=off so float arithmetic is not fused, like x86-64 gcc):
//   ML=reference/repos/MeshLib
//   clang++ -O2 -std=c++11 -ffp-contract=off -I$ML/include -I$ML/include/Util \
//     scripts/oracles/harness_curvature.cpp $ML/src/Mesh.cpp $ML/src/MeshIO.cpp $ML/src/MNIObjIO.cpp \
//     $ML/src/VtkIO.cpp $ML/src/Util/Geom.cpp $ML/src/Util/SurfaceUtil.cpp -o harness_curvature
// No MeshLib source is modified; the private helpers of SurfaceUtil are reached via #define private public.
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <float.h>
#define private public
#include "SurfaceUtil.h"
#undef private

static std::string OUT;
template <class T> static void dump(const std::string &name, const char *ext, const T *p, size_t n)
{
	std::string fn = OUT + "/" + name + "." + ext;
	FILE *f = fopen(fn.c_str(), "wb");
	if (!f) { fprintf(stderr, "cannot write %s\n", fn.c_str()); exit(1); }
	fwrite(p, sizeof(T), n, f);
	fclose(f);
}
static std::vector<int> parse(const char *s)
{
	std::vector<int> v; std::stringstream ss(s); std::string t;
	while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(atoi(t.c_str()));
	return v;
}
static void load(Mesh &m, const char *path) { m.openFile(path); }

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: %s in.vtk outdir [tensor_iters] [sphere_iters]\n", argv[0]); return 1; }
	OUT = argv[2];
	std::vector<int> titers = parse(argc > 3 ? argv[3] : "0,3");
	std::vector<int> siters = parse(argc > 4 ? argv[4] : "1,2,5,20,100,10000");

	Mesh mesh; load(mesh, argv[1]);
	const int n = mesh.nVertex(), nf = mesh.nFace();
	{
		std::vector<int> cnt(n), nbr;
		for (int i = 0; i < n; i++) { cnt[i] = mesh.vertex(i)->nNeighbor(); for (int j = 0; j < cnt[i]; j++) nbr.push_back(mesh.vertex(i)->list(j)); }
		dump("nbr_count", "i32", cnt.data(), n); dump("nbr", "i32", nbr.data(), nbr.size());
		std::vector<float> fn(nf * 3), vn(n * 3);
		for (int i = 0; i < nf; i++) for (int k = 0; k < 3; k++) fn[i * 3 + k] = mesh.face(i)->faceNormal().fv()[k];
		for (int i = 0; i < n; i++) for (int k = 0; k < 3; k++) vn[i * 3 + k] = mesh.normal(i)->fv()[k];
		dump("fnormal", "f32", fn.data(), fn.size()); dump("vnormal", "f32", vn.data(), vn.size());
	}

	for (size_t q = 0; q < titers.size(); q++)
	{
		int it = titers[q]; char tag[32]; sprintf(tag, "_it%d", it);
		// staged double path: tensor -> smoothingTensor -> principalCurvature
		std::vector<double> work(9 * n, 0.0); std::vector<double **> Tv(n); std::vector<double *> rows(3 * n);
		for (int i = 0; i < n; i++) { rows[3*i] = &work[i*9]; rows[3*i+1] = &work[i*9+3]; rows[3*i+2] = &work[i*9+6]; Tv[i] = &rows[3*i]; }
		SurfaceUtil::tensor(&mesh, Tv);
		if (it > 0) SurfaceUtil::smoothingTensor(&mesh, Tv, it);
		dump(std::string("T") + tag, "f64", work.data(), work.size());
		std::vector<double> cmin(n, 0), cmax(n, 0), wmin(3 * n, 0), wmax(3 * n, 0);
		std::vector<double *> umin(n), umax(n);
		for (int i = 0; i < n; i++) { umin[i] = &wmin[3 * i]; umax[i] = &wmax[3 * i]; }
		SurfaceUtil::principalCurvature(Tv, n, cmin.data(), cmax.data(), umin.data(), umax.data());
		std::vector<double> ou(3 * n), ov(3 * n);
		for (int i = 0; i < n; i++) for (int k = 0; k < 3; k++) { ou[3*i+k] = umin[i][k]; ov[3*i+k] = umax[i][k]; }
		dump(std::string("cmin") + tag, "f64", cmin.data(), n); dump(std::string("cmax") + tag, "f64", cmax.data(), n);
		dump(std::string("umin") + tag, "f64", ou.data(), 3 * n); dump(std::string("umax") + tag, "f64", ov.data(), 3 * n);
		std::vector<double> H(n), K(n);
		for (int i = 0; i < n; i++) { H[i] = (cmin[i] + cmax[i]) / 2; K[i] = cmin[i] * cmax[i]; }
		dump(std::string("mean") + tag, "f64", H.data(), n); dump(std::string("gauss") + tag, "f64", K.data(), n);

		// public double and float entry points (zero-initialised output buffers for the double one)
		std::vector<double> c1(n, 0), c2(n, 0), w1(3 * n, 0), w2(3 * n, 0); std::vector<double *> p1(n), p2(n);
		for (int i = 0; i < n; i++) { p1[i] = &w1[3 * i]; p2[i] = &w2[3 * i]; }
		SurfaceUtil::curvature(&mesh, c1.data(), c2.data(), p1.data(), p2.data(), it);
		std::vector<double> u1(3 * n), u2(3 * n);
		for (int i = 0; i < n; i++) for (int k = 0; k < 3; k++) { u1[3*i+k] = p1[i][k]; u2[3*i+k] = p2[i][k]; }
		dump(std::string("pub_cmin") + tag, "f64", c1.data(), n); dump(std::string("pub_cmax") + tag, "f64", c2.data(), n);
		dump(std::string("pub_umin") + tag, "f64", u1.data(), 3 * n); dump(std::string("pub_umax") + tag, "f64", u2.data(), 3 * n);
		std::vector<float> fc1(n), fc2(n), fw1(3 * n), fw2(3 * n); std::vector<float *> fp1(n), fp2(n);
		for (int i = 0; i < n; i++) { fp1[i] = &fw1[3 * i]; fp2[i] = &fw2[3 * i]; }
		SurfaceUtil::curvature(&mesh, fc1.data(), fc2.data(), fp1.data(), fp2.data(), it);
		dump(std::string("f_cmin") + tag, "f32", fc1.data(), n); dump(std::string("f_cmax") + tag, "f32", fc2.data(), n);
		dump(std::string("f_umin") + tag, "f32", fw1.data(), 3 * n); dump(std::string("f_umax") + tag, "f32", fw2.data(), 3 * n);
		// NULL direction pointers (default arguments)
		std::vector<float> g1(n), g2(n);
		SurfaceUtil::curvature(&mesh, g1.data(), g2.data(), (float **)NULL, (float **)NULL, it);
		dump(std::string("fnull_cmin") + tag, "f32", g1.data(), n); dump(std::string("fnull_cmax") + tag, "f32", g2.data(), n);
	}

	{ // quadratic fit
		std::vector<double> c1(n, 0), c2(n, 0), w1(3 * n, 0), w2(3 * n, 0); std::vector<double *> p1(n), p2(n);
		for (int i = 0; i < n; i++) { p1[i] = &w1[3 * i]; p2[i] = &w2[3 * i]; }
		SurfaceUtil::curvature_quad(&mesh, c1.data(), c2.data(), p1.data(), p2.data());
		dump("q_cmin", "f64", c1.data(), n); dump("q_cmax", "f64", c2.data(), n);
		dump("q_umin", "f64", w1.data(), 3 * n); dump("q_umax", "f64", w2.data(), 3 * n);
		std::vector<float> fc1(n), fc2(n), fw1(3 * n), fw2(3 * n); std::vector<float *> fp1(n), fp2(n);
		for (int i = 0; i < n; i++) { fp1[i] = &fw1[3 * i]; fp2[i] = &fw2[3 * i]; }
		SurfaceUtil::curvature_quad(&mesh, fc1.data(), fc2.data(), fp1.data(), fp2.data());
		dump("qf_cmin", "f32", fc1.data(), n); dump("qf_cmax", "f32", fc2.data(), n);
		dump("qf_umin", "f32", fw1.data(), 3 * n); dump("qf_umax", "f32", fw2.data(), 3 * n);
	}

	for (int type = 0; type < 2; type++)
		for (size_t q = 0; q < siters.size(); q++)
		{
			Mesh m; load(m, argv[1]);
			SurfaceUtil::sphere(&m, type, siters[q]);
			std::vector<float> v(3 * n);
			for (int i = 0; i < n; i++) for (int k = 0; k < 3; k++) v[3*i+k] = m.vertex(i)->fv()[k];
			char nm[64]; sprintf(nm, "sph_t%d_i%d", type, siters[q]);
			dump(nm, "f32", v.data(), v.size());
		}
	return 0;
}
