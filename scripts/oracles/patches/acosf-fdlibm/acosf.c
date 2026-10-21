/* Build-only shim: fdlibm/FreeBSD single-precision acos (the algorithm of glibc's sysdeps/ieee754/flt-32/e_acosf.c,
 * used by the Linux x86-64 image).  Linked into the oracle so that `acosf` (called by MeshLib's Mesh::updateNormal
 * on float angles) rounds like glibc instead of like Apple's libm: face-angle weights feed the vertex normals and the
 * anisotropic LGI kernels are sensitive to 1 ulp of them (docs/reports/lgi-hull-2026-10-20.md).
 * Original: Copyright (C) 1993 by Sun Microsystems, Inc. Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice is preserved. */
#include <stdint.h>
#include <string.h>
#include <math.h>

static const float one = 1.0000000000e+00f, pi = 3.1415925026e+00f, pio2_hi = 1.5707962513e+00f,
                   pio2_lo = 7.5497894159e-08f, pS0 = 1.6666667163e-01f, pS1 = -3.2556581497e-01f,
                   pS2 = 2.0121252537e-01f, pS3 = -4.0055535734e-02f, pS4 = 7.9153501429e-04f,
                   pS5 = 3.4793309169e-05f, qS1 = -2.4033949375e+00f, qS2 = 2.0209457874e+00f,
                   qS3 = -6.8828397989e-01f, qS4 = 7.7038154006e-02f;

float acosf(float x)
{
	float z, p, q, r, w, s, c, df;
	int32_t hx, ix;
	memcpy(&hx, &x, 4);
	ix = hx & 0x7fffffff;
	if (ix == 0x3f800000) {
		if (hx > 0) return 0.0f;
		return pi + 2.0f * pio2_lo;
	} else if (ix > 0x3f800000) {
		return (x - x) / (x - x);
	}
	if (ix < 0x3f000000) {
		if (ix <= 0x23000000) return pio2_hi + pio2_lo;
		z = x * x;
		p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
		q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
		r = p / q;
		return pio2_hi - (x - (pio2_lo - x * r));
	} else if (hx < 0) {
		z = (one + x) * 0.5f;
		p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
		q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
		s = sqrtf(z);
		r = p / q;
		w = r * s - pio2_lo;
		return pi - 2.0f * (s + w);
	} else {
		int32_t idf;
		z = (one - x) * 0.5f;
		s = sqrtf(z);
		df = s;
		memcpy(&idf, &df, 4);
		idf &= 0xfffff000;
		memcpy(&df, &idf, 4);
		c = (z - df * df) / (s + df);
		p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
		q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
		r = p / q;
		w = r * s + c;
		return 2.0f * (df + w);
	}
}
