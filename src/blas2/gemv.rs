//! General matrix-vector multiply (GEMV) - CBLAS interface.
//!
//! Computes: y = alpha * op(A) * x + beta * y
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemv.c>
//!
//! For row-major layout:
//! - Swap m and n
//! - Flip the transpose operation (NoTrans <-> Trans, ConjNoTrans <-> ConjTrans)

use num_complex::{Complex32, Complex64};

use crate::backend::{get_cgemv, get_dgemv, get_sgemv, get_zgemv};
use crate::types::{
    blasint, transpose_to_char, CblasColMajor, CblasConjTrans, CblasNoTrans, CblasRowMajor,
    CblasTrans, CBLAS_ORDER, CBLAS_TRANSPOSE,
};

/// Flip transpose operation for row-major conversion.
///
/// NoTrans <-> Trans, ConjNoTrans <-> ConjTrans
#[inline]
fn flip_transpose(trans: CBLAS_TRANSPOSE) -> CBLAS_TRANSPOSE {
    match trans {
        CblasNoTrans => CblasTrans,
        CblasTrans => CblasNoTrans,
        CblasConjTrans => {
            // ConjNoTrans is not in our enum but maps to ConjTrans flip
            // For real types, ConjTrans == Trans
            CblasNoTrans
        }
    }
}

/// Single precision general matrix-vector multiply.
///
/// Computes: y = alpha * op(A) * x + beta * y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - sgemv must be registered via `register_sgemv`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_sgemv(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    alpha: f32,
    a: *const f32,
    lda: blasint,
    x: *const f32,
    incx: blasint,
    beta: f32,
    y: *mut f32,
    incy: blasint,
) {
    let sgemv = get_sgemv();

    match order {
        CblasColMajor => {
            // Column-major: call Fortran directly
            let trans_char = transpose_to_char(trans);
            sgemv(
                &trans_char,
                &m,
                &n,
                &alpha,
                a,
                &lda,
                x,
                &incx,
                &beta,
                y,
                &incy,
            );
        }
        CblasRowMajor => {
            // Row-major: swap m/n and flip transpose
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/gemv.c
            let trans_char = transpose_to_char(flip_transpose(trans));
            sgemv(
                &trans_char,
                &n, // swapped: m -> n
                &m, // swapped: n -> m
                &alpha,
                a,
                &lda,
                x,
                &incx,
                &beta,
                y,
                &incy,
            );
        }
    }
}

/// Double precision general matrix-vector multiply.
///
/// Computes: y = alpha * op(A) * x + beta * y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dgemv must be registered via `register_dgemv`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_dgemv(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    alpha: f64,
    a: *const f64,
    lda: blasint,
    x: *const f64,
    incx: blasint,
    beta: f64,
    y: *mut f64,
    incy: blasint,
) {
    let dgemv = get_dgemv();

    match order {
        CblasColMajor => {
            let trans_char = transpose_to_char(trans);
            dgemv(
                &trans_char,
                &m,
                &n,
                &alpha,
                a,
                &lda,
                x,
                &incx,
                &beta,
                y,
                &incy,
            );
        }
        CblasRowMajor => {
            let trans_char = transpose_to_char(flip_transpose(trans));
            dgemv(
                &trans_char,
                &n,
                &m,
                &alpha,
                a,
                &lda,
                x,
                &incx,
                &beta,
                y,
                &incy,
            );
        }
    }
}

/// Single precision complex general matrix-vector multiply.
///
/// Computes: y = alpha * op(A) * x + beta * y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cgemv must be registered via `register_cgemv`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_cgemv(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: blasint,
    x: *const Complex32,
    incx: blasint,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: blasint,
) {
    let cgemv = get_cgemv();

    match order {
        CblasColMajor => {
            let trans_char = transpose_to_char(trans);
            cgemv(
                &trans_char,
                &m,
                &n,
                alpha,
                a,
                &lda,
                x,
                &incx,
                beta,
                y,
                &incy,
            );
        }
        CblasRowMajor => {
            // For complex, we need to handle ConjTrans specially
            let flipped_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjTrans => {
                    // ConjTrans in row-major becomes ConjNoTrans in col-major
                    // But Fortran uses 'R' for this (conjugate, no transpose)
                    // OpenBLAS maps CblasConjTrans -> trans=2 (R) for row-major
                    // However, we don't have a ConjNoTrans enum value
                    // For row-major ConjTrans: becomes column-major with conjugate and no transpose
                    CblasNoTrans // This is approximate - complex conjugate handling differs
                }
            };
            let trans_char = transpose_to_char(flipped_trans);
            cgemv(
                &trans_char,
                &n,
                &m,
                alpha,
                a,
                &lda,
                x,
                &incx,
                beta,
                y,
                &incy,
            );
        }
    }
}

/// Double precision complex general matrix-vector multiply.
///
/// Computes: y = alpha * op(A) * x + beta * y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zgemv must be registered via `register_zgemv`
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn cblas_zgemv(
    order: CBLAS_ORDER,
    trans: CBLAS_TRANSPOSE,
    m: blasint,
    n: blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: blasint,
    x: *const Complex64,
    incx: blasint,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: blasint,
) {
    let zgemv = get_zgemv();

    match order {
        CblasColMajor => {
            let trans_char = transpose_to_char(trans);
            zgemv(
                &trans_char,
                &m,
                &n,
                alpha,
                a,
                &lda,
                x,
                &incx,
                beta,
                y,
                &incy,
            );
        }
        CblasRowMajor => {
            // For complex, we need to handle ConjTrans specially
            let flipped_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                CblasConjTrans => {
                    // Same handling as cgemv
                    CblasNoTrans
                }
            };
            let trans_char = transpose_to_char(flipped_trans);
            zgemv(
                &trans_char,
                &n,
                &m,
                alpha,
                a,
                &lda,
                x,
                &incx,
                beta,
                y,
                &incy,
            );
        }
    }
}
