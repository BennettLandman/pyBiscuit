// Build-only shim: upstream HSD includes <lapacke.h>/<cblas.h> (OpenBLAS). On macOS we
// route both to Accelerate (new LAPACK interface). No numeric source changes.
#pragma once
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
