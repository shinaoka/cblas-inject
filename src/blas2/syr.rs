//! Symmetric/Hermitian rank-1 and rank-2 updates (SYR, HER, SYR2, HER2) - CBLAS interface.
//!
//! SYR:  A = alpha * x * x^T + A  (symmetric rank-1 update)
//! HER:  A = alpha * x * conj(x)^T + A  (hermitian rank-1 update)
//! SYR2: A = alpha * x * y^T + alpha * y * x^T + A  (symmetric rank-2 update)
//! HER2: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A  (hermitian rank-2 update)
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/zher.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/syr2.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/zher2.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_cher, get_cher2, get_dsyr, get_dsyr2, get_ssyr, get_ssyr2, get_zher, get_zher2,
};
use crate::types::{
    blasint, uplo_to_char, CblasColMajor, CblasLower, CblasRowMajor, CblasUpper, CBLAS_ORDER,
    CBLAS_UPLO,
};

// =============================================================================
// Real SYR: A = alpha * x * x^T + A
// =============================================================================

/// Single precision symmetric rank-1 update: A = alpha * x * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssyr must be registered via `register_ssyr`
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyr(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f32,
    x: *const f32,
    incx: blasint,
    a: *mut f32,
    lda: blasint,
) {
    let ssyr = get_ssyr();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            ssyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
        }
        CblasRowMajor => {
            // Row-major: invert uplo (Upper <-> Lower)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            ssyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
        }
    }
}

/// Double precision symmetric rank-1 update: A = alpha * x * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyr must be registered via `register_dsyr`
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyr(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f64,
    x: *const f64,
    incx: blasint,
    a: *mut f64,
    lda: blasint,
) {
    let dsyr = get_dsyr();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            dsyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
        }
        CblasRowMajor => {
            // Row-major: invert uplo (Upper <-> Lower)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            dsyr(&uplo_char, &n, &alpha, x, &incx, a, &lda);
        }
    }
}

// =============================================================================
// Complex HER: A = alpha * x * conj(x)^T + A
// =============================================================================

/// Single precision complex hermitian rank-1 update: A = alpha * x * conj(x)^T + A
///
/// Note: alpha is real for HER operations.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cher must be registered via `register_cher`
#[no_mangle]
pub unsafe extern "C" fn cblas_cher(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f32,
    x: *const Complex32,
    incx: blasint,
    a: *mut Complex32,
    lda: blasint,
) {
    let cher = get_cher();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            cher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
        }
        CblasRowMajor => {
            // Row-major for Hermitian: invert uplo
            // The Hermitian property A = A^H means A^T = conj(A)
            // So row-major upper triangle = col-major lower triangle (conjugated)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            cher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
        }
    }
}

/// Double precision complex hermitian rank-1 update: A = alpha * x * conj(x)^T + A
///
/// Note: alpha is real for HER operations.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zher must be registered via `register_zher`
#[no_mangle]
pub unsafe extern "C" fn cblas_zher(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f64,
    x: *const Complex64,
    incx: blasint,
    a: *mut Complex64,
    lda: blasint,
) {
    let zher = get_zher();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            zher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
        }
        CblasRowMajor => {
            // Row-major for Hermitian: invert uplo
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            zher(&uplo_char, &n, &alpha, x, &incx, a, &lda);
        }
    }
}

// =============================================================================
// Real SYR2: A = alpha * x * y^T + alpha * y * x^T + A
// =============================================================================

/// Single precision symmetric rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - ssyr2 must be registered via `register_ssyr2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_ssyr2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f32,
    x: *const f32,
    incx: blasint,
    y: *const f32,
    incy: blasint,
    a: *mut f32,
    lda: blasint,
) {
    let ssyr2 = get_ssyr2();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            ssyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
        }
        CblasRowMajor => {
            // Row-major: invert uplo (Upper <-> Lower)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            ssyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
        }
    }
}

/// Double precision symmetric rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dsyr2 must be registered via `register_dsyr2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dsyr2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f64,
    x: *const f64,
    incx: blasint,
    y: *const f64,
    incy: blasint,
    a: *mut f64,
    lda: blasint,
) {
    let dsyr2 = get_dsyr2();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            dsyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
        }
        CblasRowMajor => {
            // Row-major: invert uplo (Upper <-> Lower)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            dsyr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
        }
    }
}

// =============================================================================
// Complex HER2: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
// =============================================================================

/// Single precision complex hermitian rank-2 update
///
/// Computes: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cher2 must be registered via `register_cher2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cher2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: blasint,
    y: *const Complex32,
    incy: blasint,
    a: *mut Complex32,
    lda: blasint,
) {
    let cher2 = get_cher2();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            cher2(&uplo_char, &n, alpha, x, &incx, y, &incy, a, &lda);
        }
        CblasRowMajor => {
            // Row-major for Hermitian: invert uplo
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            // For HER2 in row-major, we also need to swap x and y
            // and use conjugate of alpha (handled by the property of HER2)
            cher2(&uplo_char, &n, alpha, y, &incy, x, &incx, a, &lda);
        }
    }
}

/// Double precision complex hermitian rank-2 update
///
/// Computes: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zher2 must be registered via `register_zher2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zher2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: blasint,
    y: *const Complex64,
    incy: blasint,
    a: *mut Complex64,
    lda: blasint,
) {
    let zher2 = get_zher2();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            zher2(&uplo_char, &n, alpha, x, &incx, y, &incy, a, &lda);
        }
        CblasRowMajor => {
            // Row-major for Hermitian: invert uplo and swap x<->y
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            zher2(&uplo_char, &n, alpha, y, &incy, x, &incx, a, &lda);
        }
    }
}
