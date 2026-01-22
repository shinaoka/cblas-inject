/*
 * Minimal common.h for cblas-trampoline ctest
 *
 * This replaces OpenBLAS's complex common.h with only the definitions
 * needed for the CBLAS test suite.
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Integer type for BLAS operations */
#ifdef USE64BITINT
typedef int64_t blasint;
#define blasabs(x) labs(x)
#else
typedef int32_t blasint;
#define blasabs(x) abs(x)
#endif

/* Fortran name mangling - we use ADD_ (trailing underscore) by default */
#ifndef BLASFUNC
#ifdef ADD_
#define BLASFUNC(x) x##_
#elif defined(UPCASE)
#define BLASFUNC(x) X
#else
#define BLASFUNC(x) x
#endif
#endif

/* Constants */
#define OPENBLAS_CONST const

#ifdef __cplusplus
}
#endif

#endif /* COMMON_H */
