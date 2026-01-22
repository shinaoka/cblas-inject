//! Hermitian rank-k update (HERK) - CBLAS interface.
//!
//! Computes: C = alpha * A * A^H + beta * C  (Trans=NoTrans)
//!       or: C = alpha * A^H * A + beta * C  (Trans=ConjTrans)
//! where C is Hermitian.
//!
//! Note: alpha and beta are REAL (not complex) for HERK operations.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/herk.c>

use crate::backend::{get_cherk, get_zherk};
use crate::types::{
    blasint, transpose_to_char, uplo_to_char, CblasColMajor, CblasConjTrans, CblasLower,
    CblasNoTrans, CblasRowMajor, CblasUpper, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Single precision complex Hermitian rank-k update.
///
/// Computes: C = alpha * A * A^H + beta * C  (Trans=NoTrans)
///       or: C = alpha * A^H * A + beta * C  (Trans=ConjTrans)
///
/// Note: alpha and beta are real (f32), not complex.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cherk must be registered via `register_cherk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cherk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: blasint,
    k: blasint,
    alpha: f32,
    a: *const Complex32,
    lda: blasint,
    beta: f32,
    c: *mut Complex32,
    ldc: blasint,
) {
    let cherk = get_cherk();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            cherk(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                &beta,
                c,
                &ldc,
            );
        }
        CblasRowMajor => {
            // Row-major: invert trans (NoTrans<->ConjTrans), invert uplo
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/herk.c
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasConjTrans,
                CblasConjTrans => CblasNoTrans,
                // Trans is not valid for HERK, but handle it like NoTrans
                _ => CblasConjTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            cherk(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                &beta,
                c,
                &ldc,
            );
        }
    }
}

/// Double precision complex Hermitian rank-k update.
///
/// Computes: C = alpha * A * A^H + beta * C  (Trans=NoTrans)
///       or: C = alpha * A^H * A + beta * C  (Trans=ConjTrans)
///
/// Note: alpha and beta are real (f64), not complex.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zherk must be registered via `register_zherk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zherk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: blasint,
    k: blasint,
    alpha: f64,
    a: *const Complex64,
    lda: blasint,
    beta: f64,
    c: *mut Complex64,
    ldc: blasint,
) {
    let zherk = get_zherk();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            zherk(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                &beta,
                c,
                &ldc,
            );
        }
        CblasRowMajor => {
            // Row-major: invert trans (NoTrans<->ConjTrans), invert uplo
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasConjTrans,
                CblasConjTrans => CblasNoTrans,
                // Trans is not valid for HERK, but handle it like NoTrans
                _ => CblasConjTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            zherk(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                &alpha,
                a,
                &lda,
                &beta,
                c,
                &ldc,
            );
        }
    }
}
