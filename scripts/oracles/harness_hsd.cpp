// Oracle harness for HSD internals (upstream f169595, v1.3.9).
//
// Constructs HSD exactly like wrapper/main.cpp (same CLI flags, minus the *Dir/--filter forms that are
// commented out upstream) and, instead of run(), drives private members (the class is opened with
// `#define private public`; the layout does not change) and dumps state as .npy files.
//
// Extra control comes from environment variables (CLI11 would reject unknown flags):
//   HSD_H_MODE   init | eval | guess | run               (default init)
//   HSD_H_OUT    output directory (must exist)
//   eval mode:
//     HSD_H_COEFF_A  raw little-endian float64 file, layout [subject][component 0..2][(degree+1)^2]
//     HSD_H_COEFF_B  optional second state, evaluated with cost() after the gradient
//     HSD_H_DEGINC   m_degree_inc (default 0)
//     HSD_H_BEG, HSD_H_END   deg_beg, deg_end of the gradient (default DEGINC, DEGINC)
//     HSD_H_LAMBDA   LM damping passed to updateGradient (default 0.001)
//     HSD_H_NSUCC    nSuccessIter (default 0)
//     HSD_H_TRANS    1: phase transition at A as in optimization() (flipping init, displacement, property
//                    statistics); 2: only updatePropertyStats()
// Dumps: <tag>.<name>.npy with name in {samples, ico_faces, s<i>.vertex, s<i>.p0, s<i>.property,
//   s<i>.tree_cache, s<i>.area0, s<i>.area1, s<i>.coeff, s<i>.gradient, s<i>.hessian, s<i>.pole, s<i>.tan,
//   feature, landmark (deformed landmarks, only with -l), mean, variance, energy, cost_lm}.
// Landmarks / location features (-l, --surface, --locationWeight, --weightMap) are passed as in main.cpp; they
// only parse when built with HSD_PATCH_LANDMARKS=1 (otherwise the lists stay empty and NULL is passed).
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "PARSE_ARGS.h"
#define private public
#include "HSD.h"
#undef private

static std::string OUT;

template <class T> static const char *descr();
template <> const char *descr<float>() { return "<f4"; }
template <> const char *descr<double>() { return "<f8"; }
template <> const char *descr<int>() { return "<i4"; }

template <class T>
static void npy(const std::string &name, const T *data, std::vector<long> shape)
{
	std::string path = OUT + "/" + name + ".npy";
	std::ostringstream sh;
	sh << "(";
	long n = 1;
	for (size_t i = 0; i < shape.size(); i++) { sh << shape[i] << (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : "")); n *= shape[i]; }
	sh << ")";
	std::string hdr = std::string("{'descr': '") + descr<T>() + "', 'fortran_order': False, 'shape': " + sh.str() + ", }";
	size_t total = 10 + hdr.size() + 1;
	size_t pad = (64 - total % 64) % 64;
	hdr += std::string(pad, ' ') + "\n";
	FILE *fp = fopen(path.c_str(), "wb");
	if (!fp) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
	fwrite("\x93NUMPY\x01\x00", 1, 8, fp);
	unsigned short hl = (unsigned short)hdr.size();
	fwrite(&hl, 2, 1, fp);
	fwrite(hdr.data(), 1, hdr.size(), fp);
	fwrite(data, sizeof(T), n, fp);
	fclose(fp);
}

static const char *env(const char *k, const char *d) { const char *v = getenv(k); return v ? v : d; }

static std::vector<double> loadDoubles(const char *path, size_t n)
{
	std::vector<double> v(n);
	FILE *fp = fopen(path, "rb");
	if (!fp || fread(v.data(), sizeof(double), n, fp) != n) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
	fclose(fp);
	return v;
}

static void dumpState(HSD &h, const std::string &tag, bool withEnergy)
{
	int csize = (h.m_degree + 1) * (h.m_degree + 1);
	int nProp = h.m_nProperties + h.m_nSurfaceProperties;
	int nLm = h.m_spharm[0].landmark.size();
	int stride = nLm * 3 + h.m_nSamples * nProp;
	for (int s = 0; s < h.m_nSubj; s++)
	{
		std::string p = tag + ".s" + std::to_string(s);
		int nv = h.m_spharm[s].sphere->nVertex();
		std::vector<float> v(nv * 3), p0(nv * 3);
		for (int i = 0; i < nv; i++)
		{
			memcpy(&v[i * 3], h.m_spharm[s].sphere->vertex(i)->fv(), 12);
			memcpy(&p0[i * 3], h.m_spharm[s].vertex[i]->p0, 12);
		}
		npy(p + ".vertex", v.data(), {nv, 3});
		npy(p + ".p0", p0.data(), {nv, 3});
		npy(p + ".property", h.m_spharm[s].property, {nProp, nv});
		if (h.m_spharm[s].tree_cache) npy(p + ".tree_cache", h.m_spharm[s].tree_cache, {h.m_nQuerySamples});
		int nf = h.m_spharm[s].sphere->nFace();
		npy(p + ".area0", h.m_spharm[s].area0, {nf});
		npy(p + ".area1", h.m_spharm[s].area1, {nf});
		npy(p + ".coeff", h.m_spharm[s].coeff, {3, csize});
		npy(p + ".gradient", h.m_spharm[s].gradient, {3, csize});
		npy(p + ".pole", h.m_spharm[s].pole, {2});
		npy(p + ".tan", h.m_spharm[s].tan1, {2, 3});
	}
	std::vector<float> feat((size_t)h.m_nSubj * nProp * h.m_nSamples), lm((size_t)h.m_nSubj * nLm * 3);
	for (int s = 0; s < h.m_nSubj; s++)
	{
		memcpy(&feat[(size_t)s * nProp * h.m_nSamples], &h.m_feature[(size_t)s * stride + nLm * 3], sizeof(float) * nProp * h.m_nSamples);
		memcpy(&lm[(size_t)s * nLm * 3], &h.m_feature[(size_t)s * stride], sizeof(float) * nLm * 3);
	}
	npy(tag + ".feature", feat.data(), {h.m_nSubj, nProp, h.m_nSamples});
	if (nLm) npy(tag + ".landmark", lm.data(), {h.m_nSubj, nLm, 3});
	npy(tag + ".cost_lm", &h.m_cost_lm, {1});
	npy(tag + ".mean", h.m_mean, {nProp, h.m_nSamples});
	npy(tag + ".variance", h.m_variance, {nProp, h.m_nSamples});
	if (withEnergy)
	{
		std::vector<double> e = {0, h.m_cost_prop, h.m_cost_area, h.m_cost_disp};
		for (int s = 0; s < h.m_nSubj; s++) e.push_back(h.m_spharm[s].isFlip ? 1 : 0);
		npy(tag + ".energy", e.data(), {(long)e.size()});
	}
}

static void dumpEnergy(HSD &h, const std::string &tag, double E)
{
	std::vector<double> e = {E, h.m_cost_prop, h.m_cost_area, h.m_cost_disp};
	for (int s = 0; s < h.m_nSubj; s++) e.push_back(h.m_spharm[s].isFlip ? 1 : 0);
	npy(tag + ".energy", e.data(), {(long)e.size()});
}

int main(int argc, char **argv)
{
	PARSE_ARGS(argc, argv);
	int nSubj = listSphere.size();
	if (listWeight.empty())
		for (int i = 0; i < (int)listProperty.size() / nSubj; i++) listWeight.push_back(1);
	int nProperties = listProperty.size(), nOutputCoeff = listOutputCoeff.size(), nOutput = listOutput.size();
	int nWeight = listWeight.size(), nCoeff = listCoeff.size(), nFixed = listFixedSubj.size();
	const char **property = new const char *[nProperties];
	const char **sphere = new const char *[nSubj];
	const char **outputcoeff = nOutputCoeff ? new const char *[nOutputCoeff] : NULL;
	const char **output = nOutput ? new const char *[nOutput] : NULL;
	const char **coeff = nCoeff ? new const char *[nCoeff] : NULL;
	bool *fixedSubj = NULL;
	if (nFixed) { fixedSubj = new bool[nSubj]; memset(fixedSubj, 0, nSubj); }
	const char *prior = tmpVariance.empty() ? NULL : tmpVariance.c_str();
	const char *icomesh = icoMesh.empty() ? NULL : icoMesh.c_str();
	float *weight = new float[nWeight];
	for (int i = 0; i < nSubj; i++) sphere[i] = listSphere[i].c_str();
	for (int i = 0; i < nProperties; i++) property[i] = listProperty[i].c_str();
	for (int i = 0; i < nOutputCoeff; i++) outputcoeff[i] = listOutputCoeff[i].c_str();
	for (int i = 0; i < nOutput; i++) output[i] = listOutput[i].c_str();
	for (int i = 0; i < nCoeff; i++) coeff[i] = listCoeff[i].c_str();
	for (int i = 0; i < nWeight; i++) weight[i] = listWeight[i];
	for (int i = 0; i < nFixed; i++) fixedSubj[listFixedSubj[i]] = true;

	int nLandmark = listLandmark.size(), nSurf = listSurf.size();
	const char **landmark = nLandmark ? new const char *[nLandmark] : NULL;
	const char **surf = nSurf ? new const char *[nSurf] : NULL;
	for (int i = 0; i < nLandmark; i++) landmark[i] = listLandmark[i].c_str();
	for (int i = 0; i < nSurf; i++) surf[i] = listSurf[i].c_str();
	if (surf == NULL) weightLoc = 0;
	HSD hsd(sphere, nSubj, property, nProperties / nSubj, output, outputcoeff, weight, degree, landmark, weightMap, weightLoc, idprior, coeff, surf, maxIter, fixedSubj, icosa, realtimeCoeff, prior, !noguess, guessRes, icomesh, nCThreads, resampling);

	std::string mode = env("HSD_H_MODE", "init");
	OUT = env("HSD_H_OUT", ".");
	HSD &h = hsd;
	int csize = (h.m_degree + 1) * (h.m_degree + 1);
	int nSamples = h.m_nQuerySamples;
	npy("samples", &h.m_propertySamples[0], {nSamples, 3});
	{
		int nf = h.m_ico_mesh->nFace();
		std::vector<int> f(nf * 3);
		for (int i = 0; i < nf; i++) memcpy(&f[i * 3], h.m_ico_mesh->face(i)->list(), 12);
		npy("ico_faces", f.data(), {nf, 3});
	}

	if (mode == "init")
	{
		dumpState(h, "init", false);
	}
	else if (mode == "eval")
	{
		std::vector<double> A = loadDoubles(getenv("HSD_H_COEFF_A"), (size_t)nSubj * 3 * csize);
		int deginc = atoi(env("HSD_H_DEGINC", "0"));
		int beg = atoi(env("HSD_H_BEG", std::to_string(deginc).c_str()));
		int end = atoi(env("HSD_H_END", std::to_string(deginc).c_str()));
		double lambda = atof(env("HSD_H_LAMBDA", "0.001"));
		int trans = atoi(env("HSD_H_TRANS", "0"));
		h.nSuccessIter = atoi(env("HSD_H_NSUCC", "0"));
		h.nIter = 0;
		for (int s = 0; s < nSubj; s++) h.m_spharm[s].step = 1;
		h.m_degree_inc = deginc;
		memcpy(h.m_coeff, A.data(), sizeof(double) * nSubj * 3 * csize);
		if (trans == 1)
		{
			for (int s = 0; s < nSubj; s++) h.updateDeformation(s, true);
			h.updateProperties(-1);
			for (int s = 0; s < nSubj; s++) h.initTriangleFlipping(s);
			h.updateDisplacement(-1, h.m_degree);
			h.updatePropertyStats();
		}
		else if (trans == 2) h.updatePropertyStats();
		double E = h.cost(-1);
		dumpState(h, "A", false);
		dumpEnergy(h, "A", E);

		// updateGradient() body, split so that the Hessian can be dumped before updateNewGradient()
		bool fine = h.m_icosahedron >= h.m_fine_res;
		int n = (end + 1) * (end + 1), n0 = beg * beg, size = n - n0;
		memset(h.m_gradient, 0, sizeof(double) * h.m_csize * 3);
		for (int s = 0; s < nSubj; s++)
		{
			if (h.m_spharm[s].fixed || h.m_spharm[s].step == 0) continue;
			memset(&h.m_Hessian[csize * 3 * csize * 3 * s], 0, sizeof(double) * size * size * 3 * 3);
			if (h.m_spharm[0].landmark.size() > 0) h.updateGradientLandmark(beg, end, s);
			h.updateGradientProperties(beg, end, s);
			if (h.m_degree_inc > 0 && fine) h.updateGradientDisplacement(beg, end, s);
		}
		for (int s = 0; s < nSubj; s++)
		{
			if (h.m_spharm[s].fixed) continue;
			npy("A.s" + std::to_string(s) + ".gradient_raw", h.m_spharm[s].gradient, {3, csize});
			if (fine) npy("A.s" + std::to_string(s) + ".hessian", &h.m_Hessian[csize * 3 * csize * 3 * s], {size * 3, size * 3});
		}
		if (fine)
			for (int s = 0; s < nSubj; s++)
			{
				if (h.m_spharm[s].fixed || h.m_spharm[s].step == 0) continue;
				h.updateNewGradient(beg, end, lambda, s);
			}
		for (int s = 0; s < nSubj; s++)
			if (!h.m_spharm[s].fixed) npy("A.s" + std::to_string(s) + ".gradient_dir", h.m_spharm[s].gradient, {3, csize});

		if (getenv("HSD_H_COEFF_B"))
		{
			std::vector<double> B = loadDoubles(getenv("HSD_H_COEFF_B"), (size_t)nSubj * 3 * csize);
			memcpy(h.m_coeff, B.data(), sizeof(double) * nSubj * 3 * csize);
			double EB = h.cost(-1);
			dumpState(h, "B", false);
			dumpEnergy(h, "B", EB);
		}
	}
	else if (mode == "guess")
	{
		h.guessInitCoeff();
		dumpState(h, "guess", true);
		npy("guess.coeff_all", h.m_coeff, {nSubj, 3, csize});
	}
	else if (mode == "run")
	{
		h.run();
		for (int s = 0; s < nSubj; s++) npy("run.s" + std::to_string(s) + ".coeff", h.m_spharm[s].coeff, {3, csize});
	}
	else { fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 2; }
	return 0;
}
