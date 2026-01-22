//! Symmetric rank-2k update (SYR2K) - CBLAS interface.
//!
//! Computes: C = alpha * A * B^T + alpha * B * A^T + beta * C  (Trans=NoTrans)
//!       or: C = alpha * A^T * B + alpha * B^T * A + beta * C  (Trans=Trans)
//! where C is symmetric.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr2k.c>

use crate::backend::get_dsyr2k;
use crate::types::{
    blasint, transpose_to_char, uplo_to_char, CblasColMajor, CblasLower, CblasNoTrans,
    CblasRowMajor, CblasTrans, CblasUpper, CBLAS_ORDER, CBLAS_TRANSPOSE, CBLAS_UPLO,
};

/// Double precision symmetric rank-2k update.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyr2k must be registered via `register_dsyr2k`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyr2k(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
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
    let dsyr2k = get_dsyr2k();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            dsyr2k(
                &uplo_char,
                &trans_char,
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
            // Row-major: invert trans, invert uplo (same as syrk)
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr2k.c
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let new_trans = match trans {
                CblasNoTrans => CblasTrans,
                CblasTrans => CblasNoTrans,
                _ => CblasNoTrans,
            };
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(new_trans);
            dsyr2k(
                &uplo_char,
                &trans_char,
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
    }
}
