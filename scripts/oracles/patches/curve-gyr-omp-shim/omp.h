// Build-only serial stub for <omp.h> (Apple clang has no OpenMP). Upstream CurveExtraction/
// Gyrification include <omp.h> unguarded; with OpenMP off the pragmas are ignored and these
// stubs make the code run single-threaded (identical to OMP_NUM_THREADS=1).
#pragma once
#include <chrono>
static inline int omp_get_thread_num() { return 0; }
static inline int omp_get_num_threads() { return 1; }
static inline int omp_get_max_threads() { return 1; }
static inline void omp_set_num_threads(int) {}
static inline double omp_get_wtime()
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
