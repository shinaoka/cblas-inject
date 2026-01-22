/*
 * cblas.h for cblas-trampoline ctest
 *
 * Declares CBLAS functions provided by cblas-trampoline.
 * This header is compatible with the standard CBLAS interface.
 */

#ifndef CBLAS_H
#define CBLAS_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* CBLAS enums - must match cblas-trampoline's types.rs */
enum CBLAS_ORDER {
    CblasRowMajor = 101,
    CblasColMajor = 102
};

enum CBLAS_TRANSPOSE {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113
};

enum CBLAS_UPLO {
    CblasUpper = 121,
    CblasLower = 122
};

enum CBLAS_DIAG {
    CblasNonUnit = 131,
    CblasUnit = 132
};

enum CBLAS_SIDE {
    CblasLeft = 141,
    CblasRight = 142
};

/*
 * ==========================================================================
 * Level 3 BLAS - provided by cblas-trampoline
 * ==========================================================================
 */

/* Double precision */
void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N,
                 const blasint K, const double alpha, const double *A,
                 const blasint lda, const double *B, const blasint ldb,
                 const double beta, double *C, const blasint ldc);

void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const blasint M, const blasint N,
                 const double alpha, const double *A, const blasint lda,
                 const double *B, const blasint ldb, const double beta,
                 double *C, const blasint ldc);

void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const blasint N, const blasint K,
                 const double alpha, const double *A, const blasint lda,
                 const double beta, double *C, const blasint ldc);

void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const blasint N, const blasint K,
                  const double alpha, const double *A, const blasint lda,
                  const double *B, const blasint ldb, const double beta,
                  double *C, const blasint ldc);

void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N,
                 const double alpha, const double *A, const blasint lda,
                 double *B, const blasint ldb);

void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const blasint M, const blasint N,
                 const double alpha, const double *A, const blasint lda,
                 double *B, const blasint ldb);

/* Single precision */
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N,
                 const blasint K, const float alpha, const float *A,
                 const blasint lda, const float *B, const blasint ldb,
                 const float beta, float *C, const blasint ldc);

/* Complex double precision */
void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N,
                 const blasint K, const void *alpha, const void *A,
                 const blasint lda, const void *B, const blasint ldb,
                 const void *beta, void *C, const blasint ldc);

/* Complex single precision */
void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const blasint M, const blasint N,
                 const blasint K, const void *alpha, const void *A,
                 const blasint lda, const void *B, const blasint ldb,
                 const void *beta, void *C, const blasint ldc);

/*
 * Error handler - called by CBLAS when invalid parameters are detected
 */
void cblas_xerbla(blasint info, const char *rout, const char *form, ...);

#ifdef __cplusplus
}
#endif

#endif /* CBLAS_H */
