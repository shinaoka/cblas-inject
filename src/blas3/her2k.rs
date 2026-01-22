//! Hermitian rank-2k update (HER2K) - CBLAS interface.
//!
//! Computes: C = alpha * A * B^H + conj(alpha) * B * A^H + beta * C  (Trans=NoTrans)
//!       or: C = alpha * A^H * B + conj(alpha) * B^H * A + beta * C  (Trans=ConjTrans)
//! where C is Hermitian.
//!
//! Note: For HER2K, alpha is complex but beta is REAL.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/her2k.c>

use crate::backend::{get_cher2k, get_zher2k};
use crate::types::{
    blasint, transpose_to_char, uplo_to_char, CblasColMajor, CblasConjTrans, CblasLower,
    CblasNoTrans, CblasRowMajor, CblasUpper, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Single precision complex Hermitian rank-2k update.
///
/// Note: alpha is complex, but beta is real (f32).
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cher2k must be registered via `register_cher2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cher2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: blasint,
    k: blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: blasint,
    b: *const Complex32,
    ldb: blasint,
    beta: f32,
    c: *mut Complex32,
    ldc: blasint,
) {
    let cher2k = get_cher2k();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            cher2k(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                alpha,
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
            // Row-major: invert uplo, invert trans (NoTrans<->ConjTrans), and conjugate alpha
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/her2k.c
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasConjTrans,
                CblasConjTrans => CblasNoTrans,
                _ => CblasNoTrans,
            };
            // Conjugate alpha for row-major
            let alpha_val = *alpha;
            let conj_alpha = Complex32::new(alpha_val.re, -alpha_val.im);
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            cher2k(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                &conj_alpha,
                a,
                &lda,
                b,
                &ldb,
                &beta,
                c,
                &ldc,
            );
        }
    }
}

/// Double precision complex Hermitian rank-2k update.
///
/// Note: alpha is complex, but beta is real (f64).
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zher2k must be registered via `register_zher2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zher2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: blasint,
    k: blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: blasint,
    b: *const Complex64,
    ldb: blasint,
    beta: f64,
    c: *mut Complex64,
    ldc: blasint,
) {
    let zher2k = get_zher2k();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            zher2k(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                alpha,
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
            // Row-major: invert uplo, invert trans (NoTrans<->ConjTrans), and conjugate alpha
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/her2k.c
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasConjTrans,
                CblasConjTrans => CblasNoTrans,
                _ => CblasNoTrans,
            };
            // Conjugate alpha for row-major
            let alpha_val = *alpha;
            let conj_alpha = Complex64::new(alpha_val.re, -alpha_val.im);
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            zher2k(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                &conj_alpha,
                a,
                &lda,
                b,
                &ldb,
                &beta,
                c,
                &ldc,
            );
        }
    }
}
