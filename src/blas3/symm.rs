//! Symmetric matrix multiply (SYMM) - CBLAS interface.
//!
//! Computes: C = alpha * A * B + beta * C  (Side=Left)
//!       or: C = alpha * B * A + beta * C  (Side=Right)
//! where A is symmetric.
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/symm.c>

use crate::backend::{get_csymm, get_dsymm, get_ssymm, get_zsymm};
use crate::types::{
    blasint, side_to_char, uplo_to_char, CblasColMajor, CblasLeft, CblasLower, CblasRight,
    CblasRowMajor, CblasUpper, CBLAS_ORDER, CBLAS_SIDE, CBLAS_UPLO,
};
use num_complex::{Complex32, Complex64};

/// Double precision symmetric matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsymm must be registered via `register_dsymm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsymm(
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
                &side_char, &uplo_char, &n, // swapped
                &m, // swapped
                &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
            );
        }
    }
}

/// Single precision symmetric matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssymm must be registered via `register_ssymm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssymm(
    order: CBLAS_ORDER,
    side: CBLAS_SIDE,
    uplo: CBLAS_UPLO,
    m: blasint,
    n: blasint,
    alpha: f32,
    a: *const f32,
    lda: blasint,
    b: *const f32,
    ldb: blasint,
    beta: f32,
    c: *mut f32,
    ldc: blasint,
) {
    let ssymm = get_ssymm();

    match order {
        CblasColMajor => {
            let side_char = side_to_char(side);
            let uplo_char = uplo_to_char(uplo);
            ssymm(
                &side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
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
            ssymm(
                &side_char, &uplo_char, &n, // swapped
                &m, // swapped
                &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
            );
        }
    }
}

/// Single precision complex symmetric matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - csymm must be registered via `register_csymm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_csymm(
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
    let csymm = get_csymm();

    match order {
        CblasColMajor => {
            let side_char = side_to_char(side);
            let uplo_char = uplo_to_char(uplo);
            csymm(
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
            csymm(
                &side_char, &uplo_char, &n, // swapped
                &m, // swapped
                alpha, a, &lda, b, &ldb, beta, c, &ldc,
            );
        }
    }
}

/// Double precision complex symmetric matrix multiply.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zsymm must be registered via `register_zsymm`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zsymm(
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
    let zsymm = get_zsymm();

    match order {
        CblasColMajor => {
            let side_char = side_to_char(side);
            let uplo_char = uplo_to_char(uplo);
            zsymm(
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
            zsymm(
                &side_char, &uplo_char, &n, // swapped
                &m, // swapped
                alpha, a, &lda, b, &ldb, beta, c, &ldc,
            );
        }
    }
}
