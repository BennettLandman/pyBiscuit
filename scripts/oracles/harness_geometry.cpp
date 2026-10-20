// Oracle driver for pybiscuit.mesh.geometry (brief 02, B2.4a). Links against MeshLib.
// usage: harness_geometry in.vtk outdir [-s ITER] [-g SIGMA ITER] [-df ITER in.f32] [-dd ITER in.f64] ...
// Dumps raw little-endian binaries into outdir (see write_* calls). No numerics here:
// everything comes from MeshLib classes, except vertexArea() which is GeodesicA::vertexArea
// (private member, body copied verbatim; GeodesicA.cpp is not linked).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "Mesh.h"
#include "Util/SurfaceUtil.h"

static std::string OUT;
static void dump(const std::string &name, const void *p, size_t bytes)
{
	FILE *f = fopen((OUT + "/" + name).c_str(), "wb");
	if (!f) { perror(name.c_str()); exit(1); }
	fwrite(p, 1, bytes, f);
	fclose(f);
}
static void dump_geometry(const Mesh &m, const std::string &pre)
{
	int n = m.nVertex(), nf = m.nFace();
	std::vector<float> vn(3 * n), fn(3 * nf), v(3 * n);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < 3; j++) { vn[3 * i + j] = m.normal(i)->fv()[j]; v[3 * i + j] = m.vertex(i)->fv()[j]; }
	for (int i = 0; i < nf; i++)
		for (int j = 0; j < 3; j++) fn[3 * i + j] = m.face(i)->faceNormal().fv()[j];
	dump(pre + "vertices.f32", v.data(), 4 * v.size());
	dump(pre + "vertex_normals.f32", vn.data(), 4 * vn.size());
	dump(pre + "face_normals.f32", fn.data(), 4 * fn.size());
}
static double vertexArea(const Mesh *mesh, int id)  // GeodesicA::vertexArea, verbatim
{
	int nn = mesh->vertex(id)->nNeighbor();
	double sum = 0;
	for (int j = 0; j < nn; j++)
	{
		int id1 = mesh->vertex(id)->list(j);
		int id2 = mesh->vertex(id)->list((j + 1) % nn);
		Vector E1 = Vector(mesh->vertex(id)->fv(), mesh->vertex(id1)->fv());
		Vector E2 = Vector(mesh->vertex(id)->fv(), mesh->vertex(id2)->fv());
		double a = E1.norm(), b = E2.norm();
		E1.unit(); E2.unit();
		double inner = E1 * E2;
		if (inner > 1) inner = 1;
		if (inner < -1) inner = -1;
		double theta = acos(inner);
		double area = a * b * sin(theta) / 2;
		sum += area / 3;
	}
	return sum;
}
static std::vector<char> slurp(const char *fn)
{
	FILE *f = fopen(fn, "rb");
	if (!f) { perror(fn); exit(1); }
	std::vector<char> b; char buf[65536]; size_t k;
	while ((k = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + k);
	fclose(f);
	return b;
}
int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: %s in.vtk outdir [ops]\n", argv[0]); return 2; }
	OUT = argv[2];
	Mesh m; m.openFile(argv[1]);
	int n = m.nVertex(), nf = m.nFace();
	dump_geometry(m, "");
	std::vector<int> ip(n + 1, 0), idx;
	for (int i = 0; i < n; i++) { for (int j = 0; j < m.vertex(i)->nNeighbor(); j++) idx.push_back(m.vertex(i)->list(j)); ip[i + 1] = idx.size(); }
	dump("nbr_indptr.i32", ip.data(), 4 * ip.size());
	dump("nbr_indices.i32", idx.data(), 4 * idx.size());
	std::vector<float> fa(nf); std::vector<double> va(n);
	for (int i = 0; i < nf; i++)  // expression of SurfaceUtil::sphere (area weight), without the /12
	{
		const int *l = m.face(i)->list();
		Vector P(m.vertex(l[0])->fv(), m.vertex(l[1])->fv()), Q(m.vertex(l[0])->fv(), m.vertex(l[2])->fv());
		fa[i] = P.cross(Q).norm() / 2;
	}
	for (int i = 0; i < n; i++) va[i] = vertexArea(&m, i);
	dump("face_areas.f32", fa.data(), 4 * fa.size());
	dump("vertex_areas.f64", va.data(), 8 * va.size());
	for (int a = 3; a < argc;)
	{
		std::string op = argv[a];
		Mesh s; s.openFile(argv[1]);
		char name[128];
		if (op == "-s")
		{
			int it = atoi(argv[a + 1]); a += 2;
			SurfaceUtil::smoothing(&s, it);
			snprintf(name, sizeof name, "smooth_i%d_", it);
			dump_geometry(s, name);
		}
		else if (op == "-g")
		{
			float sg = (float)atof(argv[a + 1]); int it = atoi(argv[a + 2]); a += 3;
			SurfaceUtil::smoothing(&s, sg, it);
			snprintf(name, sizeof name, "smoothsig_s%g_i%d_", sg, it);
			dump_geometry(s, name);
		}
		else if (op == "-df" || op == "-dd")
		{
			int it = atoi(argv[a + 1]); std::vector<char> d = slurp(argv[a + 2]); a += 3;
			if (op == "-df") { SurfaceUtil::smoothing(&s, it, (float *)d.data()); snprintf(name, sizeof name, "data_f32_i%d.f32", it); }
			else { SurfaceUtil::smoothing(&s, it, (double *)d.data()); snprintf(name, sizeof name, "data_f64_i%d.f64", it); }
			dump(name, d.data(), d.size());
		}
		else { fprintf(stderr, "bad op %s\n", op.c_str()); return 2; }
	}
	return 0;
}
