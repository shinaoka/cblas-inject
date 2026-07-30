#ifndef CBLAS_INJECT_H
#define CBLAS_INJECT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CBLAS_INJECT_STATUS_OK 0
#define CBLAS_INJECT_STATUS_NULL_POINTER 1
#define CBLAS_INJECT_STATUS_ALREADY_REGISTERED 2

/*
 * CBLAS layout and transpose values. These are prefixed to avoid conflicts
 * with cblas.h, which defines the standard Cblas* enum names.
 */
#define CBLAS_INJECT_ROW_MAJOR 101
#define CBLAS_INJECT_COL_MAJOR 102
#define CBLAS_INJECT_NO_TRANS 111
#define CBLAS_INJECT_TRANS 112
#define CBLAS_INJECT_CONJ_TRANS 113
#define CBLAS_INJECT_CONJ_NO_TRANS 114

typedef int32_t cblas_inject_blasint_lp64;
typedef int64_t cblas_inject_blasint_ilp64;

/*
 * Fortran dgemm provider signatures.
 *
 * LP64:
 *   void dgemm_(const char *transa, const char *transb,
 *               const int32_t *m, const int32_t *n, const int32_t *k,
 *               const double *alpha, const double *a, const int32_t *lda,
 *               const double *b, const int32_t *ldb,
 *               const double *beta, double *c, const int32_t *ldc);
 *
 * ILP64:
 *   void dgemm_(const char *transa, const char *transb,
 *               const int64_t *m, const int64_t *n, const int64_t *k,
 *               const double *alpha, const double *a, const int64_t *lda,
 *               const double *b, const int64_t *ldb,
 *               const double *beta, double *c, const int64_t *ldc);
 */
typedef void (*cblas_inject_dgemm_lp64_fn)(
    const char *transa,
    const char *transb,
    const cblas_inject_blasint_lp64 *m,
    const cblas_inject_blasint_lp64 *n,
    const cblas_inject_blasint_lp64 *k,
    const double *alpha,
    const double *a,
    const cblas_inject_blasint_lp64 *lda,
    const double *b,
    const cblas_inject_blasint_lp64 *ldb,
    const double *beta,
    double *c,
    const cblas_inject_blasint_lp64 *ldc);

typedef void (*cblas_inject_dgemm_ilp64_fn)(
    const char *transa,
    const char *transb,
    const cblas_inject_blasint_ilp64 *m,
    const cblas_inject_blasint_ilp64 *n,
    const cblas_inject_blasint_ilp64 *k,
    const double *alpha,
    const double *a,
    const cblas_inject_blasint_ilp64 *lda,
    const double *b,
    const cblas_inject_blasint_ilp64 *ldb,
    const double *beta,
    double *c,
    const cblas_inject_blasint_ilp64 *ldc);

typedef void (*cblas_inject_sgemm_lp64_fn)(const char *, const char *,
    const cblas_inject_blasint_lp64 *, const cblas_inject_blasint_lp64 *,
    const cblas_inject_blasint_lp64 *, const float *, const float *,
    const cblas_inject_blasint_lp64 *, const float *, const cblas_inject_blasint_lp64 *,
    const float *, float *, const cblas_inject_blasint_lp64 *);
typedef void (*cblas_inject_sgemm_ilp64_fn)(const char *, const char *,
    const cblas_inject_blasint_ilp64 *, const cblas_inject_blasint_ilp64 *,
    const cblas_inject_blasint_ilp64 *, const float *, const float *,
    const cblas_inject_blasint_ilp64 *, const float *, const cblas_inject_blasint_ilp64 *,
    const float *, float *, const cblas_inject_blasint_ilp64 *);

/* Complex values use two adjacent float values, real part first, imaginary
 * part second, matching the Fortran BLAS complex memory representation. */
typedef void (*cblas_inject_cgemm_lp64_fn)(const char *, const char *,
    const cblas_inject_blasint_lp64 *, const cblas_inject_blasint_lp64 *,
    const cblas_inject_blasint_lp64 *, const void *, const void *,
    const cblas_inject_blasint_lp64 *, const void *, const cblas_inject_blasint_lp64 *,
    const void *, void *, const cblas_inject_blasint_lp64 *);
typedef void (*cblas_inject_cgemm_ilp64_fn)(const char *, const char *,
    const cblas_inject_blasint_ilp64 *, const cblas_inject_blasint_ilp64 *,
    const cblas_inject_blasint_ilp64 *, const void *, const void *,
    const cblas_inject_blasint_ilp64 *, const void *, const cblas_inject_blasint_ilp64 *,
    const void *, void *, const cblas_inject_blasint_ilp64 *);

/*
 * Fortran zgemm provider signatures.
 *
 * Complex scalar and matrix pointers use the platform BLAS complex memory
 * layout: two adjacent double values, real part first, imaginary part second.
 *
 * LP64:
 *   void zgemm_(const char *transa, const char *transb,
 *               const int32_t *m, const int32_t *n, const int32_t *k,
 *               const void *alpha, const void *a, const int32_t *lda,
 *               const void *b, const int32_t *ldb,
 *               const void *beta, void *c, const int32_t *ldc);
 *
 * ILP64:
 *   void zgemm_(const char *transa, const char *transb,
 *               const int64_t *m, const int64_t *n, const int64_t *k,
 *               const void *alpha, const void *a, const int64_t *lda,
 *               const void *b, const int64_t *ldb,
 *               const void *beta, void *c, const int64_t *ldc);
 */
typedef void (*cblas_inject_zgemm_lp64_fn)(
    const char *transa,
    const char *transb,
    const cblas_inject_blasint_lp64 *m,
    const cblas_inject_blasint_lp64 *n,
    const cblas_inject_blasint_lp64 *k,
    const void *alpha,
    const void *a,
    const cblas_inject_blasint_lp64 *lda,
    const void *b,
    const cblas_inject_blasint_lp64 *ldb,
    const void *beta,
    void *c,
    const cblas_inject_blasint_lp64 *ldc);

typedef void (*cblas_inject_zgemm_ilp64_fn)(
    const char *transa,
    const char *transb,
    const cblas_inject_blasint_ilp64 *m,
    const cblas_inject_blasint_ilp64 *n,
    const cblas_inject_blasint_ilp64 *k,
    const void *alpha,
    const void *a,
    const cblas_inject_blasint_ilp64 *lda,
    const void *b,
    const cblas_inject_blasint_ilp64 *ldb,
    const void *beta,
    void *c,
    const cblas_inject_blasint_ilp64 *ldc);

int cblas_inject_blas_int_width(void);
int cblas_inject_supports_lp64_registration(void);
int cblas_inject_supports_ilp64_registration(void);

/*
 * Registration contract for all eight raw LP64/ILP64 s/d/c/z GEMM APIs:
 * NULL returns CBLAS_INJECT_STATUS_NULL_POINTER and registers nothing.
 * A non-NULL callback must use the exact ABI, remain valid after success, and
 * support concurrent calls. With exact-zero beta it must never read C at any
 * point and must initialize every logical C element on normal return,
 * including k == 0. Callbacks must not unwind or longjmp across this C ABI.
 */
int cblas_inject_register_dgemm_lp64(const void *dgemm);
int cblas_inject_register_dgemm_ilp64(const void *dgemm);
int cblas_inject_register_sgemm_lp64(const void *sgemm);
int cblas_inject_register_sgemm_ilp64(const void *sgemm);
int cblas_inject_register_cgemm_lp64(const void *cgemm);
int cblas_inject_register_cgemm_ilp64(const void *cgemm);
int cblas_inject_register_zgemm_lp64(const void *zgemm);
int cblas_inject_register_zgemm_ilp64(const void *zgemm);

void cblas_dgemm_64(
    int order,
    int transa,
    int transb,
    int64_t m,
    int64_t n,
    int64_t k,
    double alpha,
    const double *a,
    int64_t lda,
    const double *b,
    int64_t ldb,
    double beta,
    double *c,
    int64_t ldc);

void cblas_sgemm_64(int order, int transa, int transb, int64_t m, int64_t n,
    int64_t k, float alpha, const float *a, int64_t lda, const float *b,
    int64_t ldb, float beta, float *c, int64_t ldc);

void cblas_cgemm_64(int order, int transa, int transb, int64_t m, int64_t n,
    int64_t k, const void *alpha, const void *a, int64_t lda, const void *b,
    int64_t ldb, const void *beta, void *c, int64_t ldc);

void cblas_zgemm_64(
    int order,
    int transa,
    int transb,
    int64_t m,
    int64_t n,
    int64_t k,
    const void *alpha,
    const void *a,
    int64_t lda,
    const void *b,
    int64_t ldb,
    const void *beta,
    void *c,
    int64_t ldc);

#ifdef __cplusplus
}
#endif

#endif
