//! Triangular solve (TRSM) - CBLAS interface.
//!
//! Solves: op(A) * X = alpha * B  (Side=Left)
//!     or: X * op(A) = alpha * B  (Side=Right)
//! where A is triangular, and overwrites B with X.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trsm.c>

use crate::backend::get_dtrsm;
use crate::types::{
    blasint, diag_to_char, side_to_char, transpose_to_char, uplo_to_char, CblasColMajor, CblasLeft,
    CblasLower, CblasRight, CblasRowMajor, CblasUpper, CBLAS_DIAG, CBLAS_ORDER, CBLAS_SIDE,
    CBLAS_TRANSPOSE, CBLAS_UPLO,
};

/// Double precision triangular solve.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dtrsm must be registered via `register_dtrsm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dtrsm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    trans: CBLAS_TRANSPOSE,
    diag: CBLAS_DIAG,
    m: blasint,
    n: blasint,
    alpha: f64,
    a: *const f64,
    lda: blasint,
    b: *mut f64,
    ldb: blasint,
) {
    let dtrsm = get_dtrsm();

    match order {
        CblasColMajor => {
            let side_char = side_to_char(side);
            let uplo_char = uplo_to_char(uplo);
            let trans_char = transpose_to_char(trans);
            let diag_char = diag_to_char(diag);
            dtrsm(
                &side_char,
                &uplo_char,
                &trans_char,
                &diag_char,
                &m,
                &n,
                &alpha,
                a,
                &lda,
                b,
                &ldb,
            );
        }
        CblasRowMajor => {
            // Row-major: swap m↔n, invert side, invert uplo
            // Trans is NOT inverted
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/trsm.c
            let new_side = match side {
                CblasLeft => CblasRight,
                CblasRight => CblasLeft,
            };
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let side_char = side_to_char(new_side);
            let uplo_char = uplo_to_char(new_uplo);
            let trans_char = transpose_to_char(trans); // NOT inverted
            let diag_char = diag_to_char(diag);
            dtrsm(
                &side_char,
                &uplo_char,
                &trans_char,
                &diag_char,
                &n, // swapped
                &m, // swapped
                &alpha,
                a,
                &lda,
                b,
                &ldb,
            );
        }
    }
}
