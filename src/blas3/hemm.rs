//! Hermitian matrix multiply (HEMM) - CBLAS interface.
//!
//! Computes: C = alpha * A * B + beta * C  (Side=Left)
//!       or: C = alpha * B * A + beta * C  (Side=Right)
//! where A is Hermitian.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/hemm.c>

use crate::backend::{get_chemm, get_zhemm};
use crate::types::{
    blasint, side_to_char, uplo_to_char, CblasColMajor, CblasLeft, CblasLower, CblasRight,
    CblasRowMajor, CblasUpper, CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Single precision complex Hermitian matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - chemm must be registered via `register_chemm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chemm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: blasint,
    n: blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: blasint,
    b: *const Complex32,
    ldb: blasint,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: blasint,
) {
    let chemm = get_chemm();

    match order {
        CblasColMajor => {
            let side_char = side_to_char(side);
            let uplo_char = uplo_to_char(uplo);
            chemm(
                &side_char, &uplo_char, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc,
            );
        }
        CblasRowMajor => {
            // Row-major: swap m↔n, invert side, invert uplo
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
            chemm(
                &side_char, &uplo_char, &n, // swapped
                &m, // swapped
                alpha, a, &lda, b, &ldb, beta, c, &ldc,
            );
        }
    }
}

/// Double precision complex Hermitian matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zhemm must be registered via `register_zhemm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhemm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: blasint,
    n: blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: blasint,
    b: *const Complex64,
    ldb: blasint,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: blasint,
) {
    let zhemm = get_zhemm();

    match order {
        CblasColMajor => {
            let side_char = side_to_char(side);
            let uplo_char = uplo_to_char(uplo);
            zhemm(
                &side_char, &uplo_char, &m, &n, alpha, a, &lda, b, &ldb, beta, c, &ldc,
            );
        }
        CblasRowMajor => {
            // Row-major: swap m↔n, invert side, invert uplo
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
            zhemm(
                &side_char, &uplo_char, &n, // swapped
                &m, // swapped
                alpha, a, &lda, b, &ldb, beta, c, &ldc,
            );
        }
    }
}
