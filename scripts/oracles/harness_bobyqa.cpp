// Black-box oracle for upstream HSD src/bobyqa.h (f169595): runs min_bobyqa on fixed test functions and
// prints every evaluated point and the result with %.17g.  The header is compiled and run, not read or
// translated (licence question Q19, docs/clean-room.md); pybiscuit.hsd.bobyqa is compared against this output
// by scripts/oracles/make_golden_bobyqa.py.
//   usage: bobyqa_harness <case> <maxfun>
// Cases: quad, rosen, bound (minimum outside the box), cosf (float32 cost with ripples), sphere (the
// optimalMeanCost shape of HSD::initTangentPlane on fixed float32 landmarks).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "bobyqa.h"
#include "Geom.h"  // PI, as in HSD.cpp

struct Fn
{
	int which;
	double operator()(double *x)
	{
		double f = 0;
		if (which == 0) f = (x[0] - 0.3) * (x[0] - 0.3) + 3 * (x[1] + 0.2) * (x[1] + 0.2) + 0.5 * x[0] * x[1];
		else if (which == 1) f = 100 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (1 - x[0]) * (1 - x[0]);
		else if (which == 2) f = (x[0] - 5) * (x[0] - 5) + (x[1] + 0.5) * (x[1] + 0.5);
		else if (which == 3)
		{
			float a = (float)x[0], b = (float)x[1];
			float c = (a - 0.7f) * (a - 0.7f) + (b + 0.4f) * (b + 0.4f) + 0.05f * cosf(9.0f * a) * sinf(7.0f * b);
			f = c;
		}
		else
		{
			// sum of squared arc lengths from the pole (phi0 + x0, theta0 + x1) to fixed unit points (float32)
			static const float lm[5][3] = {{0.6f, 0.0f, 0.8f}, {0.0f, 0.6f, 0.8f}, {0.48f, 0.36f, 0.8f}, {-0.36f, 0.48f, 0.8f}, {0.0f, 0.0f, 1.0f}};
			float phi = 0.3f + (float)x[0], theta = 1.2f + (float)x[1];
			float p[3] = {cosf(theta) * cosf(phi), cosf(theta) * sinf(phi), sinf(theta)};
			float cost = 0;
			for (int i = 0; i < 5; i++)
			{
				float inner = p[0] * lm[i][0] + p[1] * lm[i][1] + p[2] * lm[i][2];
				if (inner > 1) inner = 1;
				else if (inner < -1) inner = -1;
				float a = acosf(inner);
				cost += a * a;
			}
			f = cost;
		}
		printf("eval %.17g %.17g %.17g\n", x[0], x[1], f);
		return f;
	}
};

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: %s case maxfun\n", argv[0]); return 2; }
	const char *names[] = {"quad", "rosen", "bound", "cosf", "sphere"};
	Fn fn; fn.which = -1;
	for (int i = 0; i < 5; i++) if (!strcmp(argv[1], names[i])) fn.which = i;
	if (fn.which < 0) { fprintf(stderr, "unknown case\n"); return 2; }
	int maxfun = atoi(argv[2]);
	double x[2] = {0, 0};
	if (fn.which == 1) { x[0] = -1.2; x[1] = 1; }
	double xl[2] = {-PI, -PI}, xu[2] = {PI, PI};
	int ret = min_bobyqa(2, x, fn, xl, xu, 1e-1, 1e-6, maxfun);
	printf("result %.17g %.17g %d\n", x[0], x[1], ret);
	return 0;
}
