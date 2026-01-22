//! General rank-1 update (GER) - CBLAS interface.
//!
//! Computes: A = alpha * x * y^T + A  (for real types)
//!       or: A = alpha * x * y^T + A  (GERU, unconjugated)
//!       or: A = alpha * x * conj(y)^T + A  (GERC, conjugated)
//!
//! Row-major conversion logic derived from OpenBLAS.
//! Copyright (c) 2011-2014, The OpenBLAS Project. BSD-3-Clause License.
//! <https://github.com/OpenMathLib/OpenBLAS/blob/develop/interface/ger.c>

use num_complex::{Complex32, Complex64};

use crate::backend::{get_cgerc, get_cgeru, get_dger, get_sger, get_zgerc, get_zgeru};
use crate::types::{blasint, CblasColMajor, CblasRowMajor, CBLAS_ORDER};

// =============================================================================
// Real GER: A = alpha * x * y^T + A
// =============================================================================

/// Single precision rank-1 update: A = alpha * x * y^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - sger must be registered via `register_sger`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_sger(
    order: CBLAS_ORDER,
    m: blasint,
    n: blasint,
    alpha: f32,
    x: *const f32,
    incx: blasint,
    y: *const f32,
    incy: blasint,
    a: *mut f32,
    lda: blasint,
) {
    let sger = get_sger();

    match order {
        CblasColMajor => {
            sger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
        }
        CblasRowMajor => {
            // Row-major: swap m<->n, swap x<->y, swap incx<->incy
            // A(m x n) in row-major = A^T(n x m) in col-major
            // x * y^T in row-major = y * x^T in col-major
            sger(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
        }
    }
}

/// Double precision rank-1 update: A = alpha * x * y^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - dger must be registered via `register_dger`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_dger(
    order: CBLAS_ORDER,
    m: blasint,
    n: blasint,
    alpha: f64,
    x: *const f64,
    incx: blasint,
    y: *const f64,
    incy: blasint,
    a: *mut f64,
    lda: blasint,
) {
    let dger = get_dger();

    match order {
        CblasColMajor => {
            dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
        }
        CblasRowMajor => {
            // Row-major: swap m<->n, swap x<->y, swap incx<->incy
            dger(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
        }
    }
}

// =============================================================================
// Complex GERU: A = alpha * x * y^T + A (unconjugated)
// =============================================================================

/// Single precision complex unconjugated rank-1 update: A = alpha * x * y^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cgeru must be registered via `register_cgeru`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cgeru(
    order: CBLAS_ORDER,
    m: blasint,
    n: blasint,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: blasint,
    y: *const Complex32,
    incy: blasint,
    a: *mut Complex32,
    lda: blasint,
) {
    let cgeru = get_cgeru();

    match order {
        CblasColMajor => {
            cgeru(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
        }
        CblasRowMajor => {
            // Row-major: swap m<->n, swap x<->y, swap incx<->incy
            cgeru(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
        }
    }
}

/// Double precision complex unconjugated rank-1 update: A = alpha * x * y^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zgeru must be registered via `register_zgeru`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zgeru(
    order: CBLAS_ORDER,
    m: blasint,
    n: blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: blasint,
    y: *const Complex64,
    incy: blasint,
    a: *mut Complex64,
    lda: blasint,
) {
    let zgeru = get_zgeru();

    match order {
        CblasColMajor => {
            zgeru(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
        }
        CblasRowMajor => {
            // Row-major: swap m<->n, swap x<->y, swap incx<->incy
            zgeru(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
        }
    }
}

// =============================================================================
// Complex GERC: A = alpha * x * conj(y)^T + A (conjugated)
// =============================================================================

/// Single precision complex conjugated rank-1 update: A = alpha * x * conj(y)^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - cgerc must be registered via `register_cgerc`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_cgerc(
    order: CBLAS_ORDER,
    m: blasint,
    n: blasint,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: blasint,
    y: *const Complex32,
    incy: blasint,
    a: *mut Complex32,
    lda: blasint,
) {
    let cgerc = get_cgerc();

    match order {
        CblasColMajor => {
            cgerc(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
        }
        CblasRowMajor => {
            // Row-major for GERC: swap m<->n, but also need to use conjugate of alpha
            // and swap to GERC(y, x) pattern. Following OpenBLAS logic.
            // A = alpha * x * conj(y)^T becomes A^T = conj(alpha) * conj(y) * x^H
            // which is equivalent to calling GERC with swapped vectors
            cgerc(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
        }
    }
}

/// Double precision complex conjugated rank-1 update: A = alpha * x * conj(y)^T + A
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Matrix dimensions and leading dimensions must be consistent
/// - zgerc must be registered via `register_zgerc`
#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn cblas_zgerc(
    order: CBLAS_ORDER,
    m: blasint,
    n: blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: blasint,
    y: *const Complex64,
    incy: blasint,
    a: *mut Complex64,
    lda: blasint,
) {
    let zgerc = get_zgerc();

    match order {
        CblasColMajor => {
            zgerc(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
        }
        CblasRowMajor => {
            // Row-major for GERC: swap m<->n, swap x<->y
            zgerc(&n, &m, alpha, y, &incy, x, &incx, a, &lda);
        }
    }
}
