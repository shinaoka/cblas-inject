//! Symmetric/Hermitian packed rank-1 and rank-2 updates (SPR, HPR, SPR2, HPR2) - CBLAS interface.
//!
//! SPR:  A = alpha * x * x^T + A  (symmetric packed rank-1 update)
//! HPR:  A = alpha * x * conj(x)^T + A  (hermitian packed rank-1 update)
//! SPR2: A = alpha * x * y^T + alpha * y * x^T + A  (symmetric packed rank-2 update)
//! HPR2: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A  (hermitian packed rank-2 update)
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/spr.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/zhpr.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/spr2.c>
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/zhpr2.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_chpr, get_chpr2, get_dspr, get_dspr2, get_sspr, get_sspr2, get_zhpr, get_zhpr2,
};
use crate::types::{
    blasint, uplo_to_char, CblasColMajor, CblasLower, CblasRowMajor, CblasUpper, CBLAS_ORDER,
    CBLAS_UPLO,
};

// =============================================================================
// Real SPR: A = alpha * x * x^T + A
// =============================================================================

/// Single precision symmetric packed rank-1 update: A = alpha * x * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - sspr must be registered via `register_sspr`
#[no_mangle]
pub unsafe extern "C" fn cblas_sspr(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f32,
    x: *const f32,
    incx: blasint,
    ap: *mut f32,
) {
    let sspr = get_sspr();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            sspr(&uplo_char, &n, &alpha, x, &incx, ap);
        }
        CblasRowMajor => {
            // Row-major: invert uplo (Upper <-> Lower)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            sspr(&uplo_char, &n, &alpha, x, &incx, ap);
        }
    }
}

/// Double precision symmetric packed rank-1 update: A = alpha * x * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - dspr must be registered via `register_dspr`
#[no_mangle]
pub unsafe extern "C" fn cblas_dspr(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f64,
    x: *const f64,
    incx: blasint,
    ap: *mut f64,
) {
    let dspr = get_dspr();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            dspr(&uplo_char, &n, &alpha, x, &incx, ap);
        }
        CblasRowMajor => {
            // Row-major: invert uplo (Upper <-> Lower)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            dspr(&uplo_char, &n, &alpha, x, &incx, ap);
        }
    }
}

// =============================================================================
// Complex HPR: A = alpha * x * conj(x)^T + A
// =============================================================================

/// Single precision complex hermitian packed rank-1 update: A = alpha * x * conj(x)^T + A
///
/// Note: alpha is real for HPR operations.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - chpr must be registered via `register_chpr`
#[no_mangle]
pub unsafe extern "C" fn cblas_chpr(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f32,
    x: *const Complex32,
    incx: blasint,
    ap: *mut Complex32,
) {
    let chpr = get_chpr();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            chpr(&uplo_char, &n, &alpha, x, &incx, ap);
        }
        CblasRowMajor => {
            // Row-major for Hermitian: invert uplo
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            chpr(&uplo_char, &n, &alpha, x, &incx, ap);
        }
    }
}

/// Double precision complex hermitian packed rank-1 update: A = alpha * x * conj(x)^T + A
///
/// Note: alpha is real for HPR operations.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - zhpr must be registered via `register_zhpr`
#[no_mangle]
pub unsafe extern "C" fn cblas_zhpr(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f64,
    x: *const Complex64,
    incx: blasint,
    ap: *mut Complex64,
) {
    let zhpr = get_zhpr();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            zhpr(&uplo_char, &n, &alpha, x, &incx, ap);
        }
        CblasRowMajor => {
            // Row-major for Hermitian: invert uplo
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            zhpr(&uplo_char, &n, &alpha, x, &incx, ap);
        }
    }
}

// =============================================================================
// Real SPR2: A = alpha * x * y^T + alpha * y * x^T + A
// =============================================================================

/// Single precision symmetric packed rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - sspr2 must be registered via `register_sspr2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_sspr2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f32,
    x: *const f32,
    incx: blasint,
    y: *const f32,
    incy: blasint,
    ap: *mut f32,
) {
    let sspr2 = get_sspr2();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            sspr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, ap);
        }
        CblasRowMajor => {
            // Row-major: invert uplo (Upper <-> Lower)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            sspr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, ap);
        }
    }
}

/// Double precision symmetric packed rank-2 update: A = alpha * x * y^T + alpha * y * x^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - dspr2 must be registered via `register_dspr2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dspr2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: f64,
    x: *const f64,
    incx: blasint,
    y: *const f64,
    incy: blasint,
    ap: *mut f64,
) {
    let dspr2 = get_dspr2();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            dspr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, ap);
        }
        CblasRowMajor => {
            // Row-major: invert uplo (Upper <-> Lower)
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            dspr2(&uplo_char, &n, &alpha, x, &incx, y, &incy, ap);
        }
    }
}

// =============================================================================
// Complex HPR2: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
// =============================================================================

/// Single precision complex hermitian packed rank-2 update
///
/// Computes: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - chpr2 must be registered via `register_chpr2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chpr2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: blasint,
    y: *const Complex32,
    incy: blasint,
    ap: *mut Complex32,
) {
    let chpr2 = get_chpr2();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            chpr2(&uplo_char, &n, alpha, x, &incx, y, &incy, ap);
        }
        CblasRowMajor => {
            // Row-major for Hermitian: invert uplo and swap x<->y
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            // For HPR2 in row-major, we swap x and y
            chpr2(&uplo_char, &n, alpha, y, &incy, x, &incx, ap);
        }
    }
}

/// Double precision complex hermitian packed rank-2 update
///
/// Computes: A = alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - The packed array `ap` must contain n*(n+1)/2 elements
/// - zhpr2 must be registered via `register_zhpr2`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhpr2(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: blasint,
    y: *const Complex64,
    incy: blasint,
    ap: *mut Complex64,
) {
    let zhpr2 = get_zhpr2();

    match order {
        CblasColMajor => {
            let uplo_char = uplo_to_char(uplo);
            zhpr2(&uplo_char, &n, alpha, x, &incx, y, &incy, ap);
        }
        CblasRowMajor => {
            // Row-major for Hermitian: invert uplo and swap x<->y
            let new_uplo = match uplo {
                CblasUpper => CblasLower,
                CblasLower => CblasUpper,
            };
            let uplo_char = uplo_to_char(new_uplo);
            // For HPR2 in row-major, we swap x and y
            zhpr2(&uplo_char, &n, alpha, y, &incy, x, &incx, ap);
        }
    }
}
