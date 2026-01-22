// ctest/common.h - Minimal replacement for cblas-trampoline tests
#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

// blasint type matching cblas-trampoline
#ifdef USE64BITINT
typedef int64_t blasint;
#else
typedef int32_t blasint;
#endif

// Fortran name mangling (pick one, match your BLAS library)
#define BLASFUNC(x) x##_

// OpenBLAS complex types (for cblas.h compatibility)
typedef struct { float real; float imag; } openblas_complex_float;
typedef struct { double real; double imag; } openblas_complex_double;

// OPENBLAS_CONST
#define OPENBLAS_CONST const

// bfloat16 and hfloat16 types (for cblas.h compatibility)
typedef uint16_t bfloat16;
typedef uint16_t hfloat16;

// Fortran BLAS declarations (for reference comparison in tests)
// Single precision
void srot_(const int *n, float *x, const int *incx, float *y, const int *incy, const float *c, const float *s);
void srotm_(const int *n, float *x, const int *incx, float *y, const int *incy, const float *param);
void srotg_(float *a, float *b, float *c, float *s);
void srotmg_(float *d1, float *d2, float *x1, const float *y1, float *param);
void sswap_(const int *n, float *x, const int *incx, float *y, const int *incy);
void scopy_(const int *n, const float *x, const int *incx, float *y, const int *incy);
void saxpy_(const int *n, const float *alpha, const float *x, const int *incx, float *y, const int *incy);
void sscal_(const int *n, const float *alpha, float *x, const int *incx);
float sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy);
float snrm2_(const int *n, const float *x, const int *incx);
float sasum_(const int *n, const float *x, const int *incx);
int isamax_(const int *n, const float *x, const int *incx);

// Double precision
void drot_(const int *n, double *x, const int *incx, double *y, const int *incy, const double *c, const double *s);
void drotm_(const int *n, double *x, const int *incx, double *y, const int *incy, const double *param);
void drotg_(double *a, double *b, double *c, double *s);
void drotmg_(double *d1, double *d2, double *x1, const double *y1, double *param);
void dswap_(const int *n, double *x, const int *incx, double *y, const int *incy);
void dcopy_(const int *n, const double *x, const int *incx, double *y, const int *incy);
void daxpy_(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);
void dscal_(const int *n, const double *alpha, double *x, const int *incx);
double ddot_(const int *n, const double *x, const int *incx, const double *y, const int *incy);
double dnrm2_(const int *n, const double *x, const int *incx);
double dasum_(const int *n, const double *x, const int *incx);
int idamax_(const int *n, const double *x, const int *incx);
double dsdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy);
float sdsdot_(const int *n, const float *sb, const float *x, const int *incx, const float *y, const int *incy);

// Single complex
void cswap_(const int *n, void *x, const int *incx, void *y, const int *incy);
void ccopy_(const int *n, const void *x, const int *incx, void *y, const int *incy);
void caxpy_(const int *n, const void *alpha, const void *x, const int *incx, void *y, const int *incy);
void cscal_(const int *n, const void *alpha, void *x, const int *incx);
void csscal_(const int *n, const float *alpha, void *x, const int *incx);
void cdotu_(void *ret, const int *n, const void *x, const int *incx, const void *y, const int *incy);
void cdotc_(void *ret, const int *n, const void *x, const int *incx, const void *y, const int *incy);
float scnrm2_(const int *n, const void *x, const int *incx);
float scasum_(const int *n, const void *x, const int *incx);
int icamax_(const int *n, const void *x, const int *incx);

// Double complex
void zswap_(const int *n, void *x, const int *incx, void *y, const int *incy);
void zcopy_(const int *n, const void *x, const int *incx, void *y, const int *incy);
void zaxpy_(const int *n, const void *alpha, const void *x, const int *incx, void *y, const int *incy);
void zscal_(const int *n, const void *alpha, void *x, const int *incx);
void zdscal_(const int *n, const double *alpha, void *x, const int *incx);
void zdotu_(void *ret, const int *n, const void *x, const int *incx, const void *y, const int *incy);
void zdotc_(void *ret, const int *n, const void *x, const int *incx, const void *y, const int *incy);
double dznrm2_(const int *n, const void *x, const int *incx);
double dzasum_(const int *n, const void *x, const int *incx);
int izamax_(const int *n, const void *x, const int *incx);

#endif
