//! Symmetric matrix multiply (SYMM) - CBLAS interface.
//!
//! Computes: C = alpha * A * B + beta * C  (Side=Left)
//!       or: C = alpha * B * A + beta * C  (Side=Right)
//! where A is symmetric.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/symm.c>

use crate::backend::get_dsymm;
use crate::types::{
    blasint, side_to_char, uplo_to_char, CblasColMajor, CblasLeft, CblasLower, CblasRight,
    CblasRowMajor, CblasUpper, CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO,
};

/// Double precision symmetric matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsymm must be registered via `register_dsymm`
#[allow(clippy::too_many_arguments)]
pub unsafe fn cblas_dsymm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: blasint,
    n: blasint,
    alpha: f64,
    a: *const f64,
    lda: blasint,
    b: *const f64,
    ldb: blasint,
    beta: f64,
    c: *mut f64,
    ldc: blasint,
) {
    let dsymm = get_dsymm();

    match order {
        CblasColMajor => {
            let side_char = side_to_char(side);
            let uplo_char = uplo_to_char(uplo);
            dsymm(
                &side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
            );
        }
        CblasRowMajor => {
            // Row-major: swap m↔n, invert side, invert uplo
            // Following OpenBLAS: https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/symm.c
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
            dsymm(
                &side_char,
                &uplo_char,
                &n, // swapped
                &m, // swapped
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
