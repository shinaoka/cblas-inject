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
    get_chpr_for_lp64_cblas, get_chpr_for_ilp64_cblas, ChprProvider,
    get_chpr2_for_lp64_cblas, get_chpr2_for_ilp64_cblas, Chpr2Provider,
    get_dspr_for_lp64_cblas, get_dspr_for_ilp64_cblas, DsprProvider,
    get_dspr2_for_lp64_cblas, get_dspr2_for_ilp64_cblas, Dspr2Provider,
    get_sspr_for_lp64_cblas, get_sspr_for_ilp64_cblas, SsprProvider,
    get_sspr2_for_lp64_cblas, get_sspr2_for_ilp64_cblas, Sspr2Provider,
    get_zhpr_for_lp64_cblas, get_zhpr_for_ilp64_cblas, ZhprProvider,
    get_zhpr2_for_lp64_cblas, get_zhpr2_for_ilp64_cblas, Zhpr2Provider,
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
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    ap: *mut f32,
) {
    let p = get_sspr_for_lp64_cblas();
    match p {
        SsprProvider::Lp64(sspr) => {

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
        SsprProvider::Ilp64(sspr) => {
            let n = n as i64; let incx = incx as i64;

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
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_sspr_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f32,
    x: *const f32,
    incx: i64,
    ap: *mut f32,
) {
    let p = get_sspr_for_ilp64_cblas();
    match p {
        SsprProvider::Ilp64(sspr) => {

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
        SsprProvider::Lp64(sspr) => {
            let n = n as i32; let incx = incx as i32;

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
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    ap: *mut f64,
) {
    let p = get_dspr_for_lp64_cblas();
    match p {
        DsprProvider::Lp64(dspr) => {

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
        DsprProvider::Ilp64(dspr) => {
            let n = n as i64; let incx = incx as i64;

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
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_dspr_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f64,
    x: *const f64,
    incx: i64,
    ap: *mut f64,
) {
    let p = get_dspr_for_ilp64_cblas();
    match p {
        DsprProvider::Ilp64(dspr) => {

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
        DsprProvider::Lp64(dspr) => {
            let n = n as i32; let incx = incx as i32;

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
    n: i32,
    alpha: f32,
    x: *const Complex32,
    incx: i32,
    ap: *mut Complex32,
) {
    let p = get_chpr_for_lp64_cblas();
    match p {
        ChprProvider::Lp64(chpr) => {

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
        ChprProvider::Ilp64(chpr) => {
            let n = n as i64; let incx = incx as i64;

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
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_chpr_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f32,
    x: *const Complex32,
    incx: i64,
    ap: *mut Complex32,
) {
    let p = get_chpr_for_ilp64_cblas();
    match p {
        ChprProvider::Ilp64(chpr) => {

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
        ChprProvider::Lp64(chpr) => {
            let n = n as i32; let incx = incx as i32;

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
    n: i32,
    alpha: f64,
    x: *const Complex64,
    incx: i32,
    ap: *mut Complex64,
) {
    let p = get_zhpr_for_lp64_cblas();
    match p {
        ZhprProvider::Lp64(zhpr) => {

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
        ZhprProvider::Ilp64(zhpr) => {
            let n = n as i64; let incx = incx as i64;

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
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_zhpr_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f64,
    x: *const Complex64,
    incx: i64,
    ap: *mut Complex64,
) {
    let p = get_zhpr_for_ilp64_cblas();
    match p {
        ZhprProvider::Ilp64(zhpr) => {

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
        ZhprProvider::Lp64(zhpr) => {
            let n = n as i32; let incx = incx as i32;

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
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
    ap: *mut f32,
) {
    let p = get_sspr2_for_lp64_cblas();
    match p {
        Sspr2Provider::Lp64(sspr2) => {

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
        Sspr2Provider::Ilp64(sspr2) => {
            let n = n as i64; let incx = incx as i64; let incy = incy as i64;

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
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_sspr2_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f32,
    x: *const f32,
    incx: i64,
    y: *const f32,
    incy: i64,
    ap: *mut f32,
) {
    let p = get_sspr2_for_ilp64_cblas();
    match p {
        Sspr2Provider::Ilp64(sspr2) => {

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
        Sspr2Provider::Lp64(sspr2) => {
            let n = n as i32; let incx = incx as i32; let incy = incy as i32;

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
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
    ap: *mut f64,
) {
    let p = get_dspr2_for_lp64_cblas();
    match p {
        Dspr2Provider::Lp64(dspr2) => {

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
        Dspr2Provider::Ilp64(dspr2) => {
            let n = n as i64; let incx = incx as i64; let incy = incy as i64;

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
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dspr2_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: f64,
    x: *const f64,
    incx: i64,
    y: *const f64,
    incy: i64,
    ap: *mut f64,
) {
    let p = get_dspr2_for_ilp64_cblas();
    match p {
        Dspr2Provider::Ilp64(dspr2) => {

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
        Dspr2Provider::Lp64(dspr2) => {
            let n = n as i32; let incx = incx as i32; let incy = incy as i32;

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
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    ap: *mut Complex32,
) {
    let p = get_chpr2_for_lp64_cblas();
    match p {
        Chpr2Provider::Lp64(chpr2) => {

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
        Chpr2Provider::Ilp64(chpr2) => {
            let n = n as i64; let incx = incx as i64; let incy = incy as i64;

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
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_chpr2_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i64,
    y: *const Complex32,
    incy: i64,
    ap: *mut Complex32,
) {
    let p = get_chpr2_for_ilp64_cblas();
    match p {
        Chpr2Provider::Ilp64(chpr2) => {

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
        Chpr2Provider::Lp64(chpr2) => {
            let n = n as i32; let incx = incx as i32; let incy = incy as i32;

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
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    ap: *mut Complex64,
) {
    let p = get_zhpr2_for_lp64_cblas();
    match p {
        Zhpr2Provider::Lp64(zhpr2) => {

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
        Zhpr2Provider::Ilp64(zhpr2) => {
            let n = n as i64; let incx = incx as i64; let incy = incy as i64;

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
    }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zhpr2_64(
    order: CBLAS_ORDER,
    uplo: CBLAS_UPLO,
    n: i64,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i64,
    y: *const Complex64,
    incy: i64,
    ap: *mut Complex64,
) {
    let p = get_zhpr2_for_ilp64_cblas();
    match p {
        Zhpr2Provider::Ilp64(zhpr2) => {

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
        Zhpr2Provider::Lp64(zhpr2) => {
            let n = n as i32; let incx = incx as i32; let incy = incy as i32;

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
    }
}
