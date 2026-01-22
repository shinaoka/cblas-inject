//! Symmetric rank-k update (SYRK) - CBLAS interface.
//!
//! Computes: C = alpha * A * A^T + beta * C  (Trans=NoTrans)
//!       or: C = alpha * A^T * A + beta * C  (Trans=Trans)
//! where C is symmetric.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syrk.c>

use crate::backend::{get_csyrk, get_dsyrk, get_ssyrk, get_zsyrk};
use crate::types::{
    blasint, transpose_to_char, uplo_to_char, CblasColMajor, CblasLower, CblasNoTrans,
    CblasRowMajor, CblasTrans, CblasUpper, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Double precision symmetric rank-k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyrk must be registered via `register_dsyrk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyrk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: blasint,
    k: blasint,
    alpha: f64,
    a: *const f64,
    lda: blasint,
    beta: f64,
    c: *mut f64,
    ldc: blasint,
) {
    let dsyrk = get_dsyrk();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            dsyrk(
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
            // Row-major: invert trans, invert uplo
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syrk.c
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                // ConjTrans is handled same as Trans for real symmetric
                _ => CblasNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            dsyrk(
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

/// Single precision symmetric rank-k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssyrk must be registered via `register_ssyrk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyrk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: blasint,
    k: blasint,
    alpha: f32,
    a: *const f32,
    lda: blasint,
    beta: f32,
    c: *mut f32,
    ldc: blasint,
) {
    let ssyrk = get_ssyrk();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            ssyrk(
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
            // Row-major: invert trans, invert uplo
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                // ConjTrans is handled same as Trans for real symmetric
                _ => CblasNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            ssyrk(
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

/// Single precision complex symmetric rank-k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - csyrk must be registered via `register_csyrk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_csyrk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: blasint,
    k: blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: blasint,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: blasint,
) {
    let csyrk = get_csyrk();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            csyrk(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                alpha,
                a,
                &lda,
                beta,
                c,
                &ldc,
            );
        }
        CblasRowMajor => {
            // Row-major: invert trans, invert uplo
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                // ConjTrans is handled same as Trans for complex symmetric
                _ => CblasNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            csyrk(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                alpha,
                a,
                &lda,
                beta,
                c,
                &ldc,
            );
        }
    }
}

/// Double precision complex symmetric rank-k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zsyrk must be registered via `register_zsyrk`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zsyrk(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    n: blasint,
    k: blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: blasint,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: blasint,
) {
    let zsyrk = get_zsyrk();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            zsyrk(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                alpha,
                a,
                &lda,
                beta,
                c,
                &ldc,
            );
        }
        CblasRowMajor => {
            // Row-major: invert trans, invert uplo
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                // ConjTrans is handled same as Trans for complex symmetric
                _ => CblasNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            zsyrk(
                &uplo_char,
                &trans_char,
                &n,
                &k,
                alpha,
                a,
                &lda,
                beta,
                c,
                &ldc,
            );
        }
    }
}
