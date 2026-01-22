//! Symmetric rank-k update (SYRK) - CBLAS interface.
//!
//! Computes: C = alpha * A * A^T + beta * C  (Trans=NoTrans)
//!       or: C = alpha * A^T * A + beta * C  (Trans=Trans)
//! where C is symmetric.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syrk.c>

use crate::backend::get_dsyrk;
use crate::types::{
    blasint, transpose_to_char, uplo_to_char, CblasColMajor, CblasLower, CblasNoTrans,
    CblasRowMajor, CblasTrans, CblasUpper, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};

/// Double precision symmetric rank-k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyrk must be registered via `register_dsyrk`
#[allow(clippy::too_many_arguments)]
pub unsafe fn cblas_dsyrk(
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
                &uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc,
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
                &uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc,
            );
        }
    }
}
