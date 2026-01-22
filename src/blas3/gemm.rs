//! General matrix multiply (GEMM) - CBLAS interface.
//!
//! Computes: C = alpha * op(A) * op(B) + beta * C
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemm.c>
//!
//! For row-major layout, we swap A↔B, m↔n, lda↔ldb, TransA↔TransB.
//! The transpose flags are NOT inverted, just swapped.

use num_complex::{Complex32, Complex64};

use crate::backend::{get_cgemm, get_dgemm, get_sgemm, get_zgemm};
use crate::types::{
    blasint, transpose_to_char, CblasColMajor, CblasRowMajor, CBLAS_ORDER, CBLAS_TRANSPOSE,
};

/// Double precision general matrix multiply.
///
/// Computes: C = alpha * op(A) * op(B) + beta * C
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dgemm must be registered via `register_dgemm`
#[allow(clippy::too_many_arguments)]
pub unsafe fn cblas_dgemm(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: f64,
    a: *const f64,
    lda: blasint,
    b: *const f64,
    ldb: blasint,
    beta: f64,
    c: *mut f64,
    ldc: blasint,
) {
    let dgemm = get_dgemm();

    match order {
        CblasColMajor => {
            // Column-major: call Fortran directly
            let transa_char = transpose_to_char(transa);
            let transb_char = transpose_to_char(transb);
            dgemm(
                &transa_char,
                &transb_char,
                &m,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                b,
                &ldb,
                &beta,
                c,
                &ldc,
            );
        }
        CblasRowMajor => {
            // Row-major: swap A↔B, m↔n, lda↔ldb, TransA↔TransB
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemm.c#L489-L537
            let transa_char = transpose_to_char(transb); // TransB becomes transa
            let transb_char = transpose_to_char(transa); // TransA becomes transb
            dgemm(
                &transa_char,
                &transb_char,
                &n, // swapped: m -> n
                &m, // swapped: n -> m
                &k,
                &alpha,
                b,    // swapped: a -> b
                &ldb, // swapped: lda -> ldb
                a,    // swapped: b -> a
                &lda, // swapped: ldb -> lda
                &beta,
                c,
                &ldc,
            );
        }
    }
}

/// Single precision general matrix multiply.
///
/// Computes: C = alpha * op(A) * op(B) + beta * C
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - sgemm must be registered via `register_sgemm`
#[allow(clippy::too_many_arguments)]
pub unsafe fn cblas_sgemm(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: f32,
    a: *const f32,
    lda: blasint,
    b: *const f32,
    ldb: blasint,
    beta: f32,
    c: *mut f32,
    ldc: blasint,
) {
    let sgemm = get_sgemm();

    match order {
        CblasColMajor => {
            let transa_char = transpose_to_char(transa);
            let transb_char = transpose_to_char(transb);
            sgemm(
                &transa_char,
                &transb_char,
                &m,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                b,
                &ldb,
                &beta,
                c,
                &ldc,
            );
        }
        CblasRowMajor => {
            let transa_char = transpose_to_char(transb);
            let transb_char = transpose_to_char(transa);
            sgemm(
                &transa_char,
                &transb_char,
                &n,
                &m,
                &k,
                &alpha,
                b,
                &ldb,
                a,
                &lda,
                &beta,
                c,
                &ldc,
            );
        }
    }
}

/// Double precision complex general matrix multiply.
///
/// Computes: C = alpha * op(A) * op(B) + beta * C
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zgemm must be registered via `register_zgemm`
#[allow(clippy::too_many_arguments)]
pub unsafe fn cblas_zgemm(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: Complex64,
    a: *const Complex64,
    lda: blasint,
    b: *const Complex64,
    ldb: blasint,
    beta: Complex64,
    c: *mut Complex64,
    ldc: blasint,
) {
    let zgemm = get_zgemm();

    match order {
        CblasColMajor => {
            let transa_char = transpose_to_char(transa);
            let transb_char = transpose_to_char(transb);
            zgemm(
                &transa_char,
                &transb_char,
                &m,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                b,
                &ldb,
                &beta,
                c,
                &ldc,
            );
        }
        CblasRowMajor => {
            let transa_char = transpose_to_char(transb);
            let transb_char = transpose_to_char(transa);
            zgemm(
                &transa_char,
                &transb_char,
                &n,
                &m,
                &k,
                &alpha,
                b,
                &ldb,
                a,
                &lda,
                &beta,
                c,
                &ldc,
            );
        }
    }
}

/// Single precision complex general matrix multiply.
///
/// Computes: C = alpha * op(A) * op(B) + beta * C
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cgemm must be registered via `register_cgemm`
#[allow(clippy::too_many_arguments)]
pub unsafe fn cblas_cgemm(
    order: CBLAS_ORDER,
    transa: CBLAS_TRANSPOSE,
    transb: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: Complex32,
    a: *const Complex32,
    lda: blasint,
    b: *const Complex32,
    ldb: blasint,
    beta: Complex32,
    c: *mut Complex32,
    ldc: blasint,
) {
    let cgemm = get_cgemm();

    match order {
        CblasColMajor => {
            let transa_char = transpose_to_char(transa);
            let transb_char = transpose_to_char(transb);
            cgemm(
                &transa_char,
                &transb_char,
                &m,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                b,
                &ldb,
                &beta,
                c,
                &ldc,
            );
        }
        CblasRowMajor => {
            let transa_char = transpose_to_char(transb);
            let transb_char = transpose_to_char(transa);
            cgemm(
                &transa_char,
                &transb_char,
                &n,
                &m,
                &k,
                &alpha,
                b,
                &ldb,
                a,
                &lda,
                &beta,
                c,
                &ldc,
            );
        }
    }
}
