//! BLAS Level 1: Vector operations (swap, copy, axpy, scal).
//!
//! These functions operate on vectors and do not require row-major conversion.

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_caxpy, get_ccopy, get_cscal, get_csscal, get_cswap, get_daxpy, get_dcopy, get_dscal,
    get_dswap, get_saxpy, get_scopy, get_sscal, get_sswap, get_zaxpy, get_zcopy, get_zdscal,
    get_zscal, get_zswap,
};
use crate::types::blasint;

// =============================================================================
// Vector swap (exchange x and y)
// =============================================================================

/// Single precision vector swap.
///
/// Exchanges the elements of vectors x and y.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - sswap must be registered via `register_sswap`
#[no_mangle]
pub unsafe extern "C" fn cblas_sswap(n: blasint, x: *mut f32, incx: blasint, y: *mut f32, incy: blasint) {
    let sswap = get_sswap();
    sswap(&n, x, &incx, y, &incy);
}

/// Double precision vector swap.
///
/// Exchanges the elements of vectors x and y.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - dswap must be registered via `register_dswap`
#[no_mangle]
pub unsafe extern "C" fn cblas_dswap(n: blasint, x: *mut f64, incx: blasint, y: *mut f64, incy: blasint) {
    let dswap = get_dswap();
    dswap(&n, x, &incx, y, &incy);
}

/// Single precision complex vector swap.
///
/// Exchanges the elements of vectors x and y.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - cswap must be registered via `register_cswap`
#[no_mangle]
pub unsafe extern "C" fn cblas_cswap(
    n: blasint,
    x: *mut Complex32,
    incx: blasint,
    y: *mut Complex32,
    incy: blasint,
) {
    let cswap = get_cswap();
    cswap(&n, x, &incx, y, &incy);
}

/// Double precision complex vector swap.
///
/// Exchanges the elements of vectors x and y.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - zswap must be registered via `register_zswap`
#[no_mangle]
pub unsafe extern "C" fn cblas_zswap(
    n: blasint,
    x: *mut Complex64,
    incx: blasint,
    y: *mut Complex64,
    incy: blasint,
) {
    let zswap = get_zswap();
    zswap(&n, x, &incx, y, &incy);
}

// =============================================================================
// Vector copy (y = x)
// =============================================================================

/// Single precision vector copy.
///
/// Copies vector x to vector y.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - scopy must be registered via `register_scopy`
#[no_mangle]
pub unsafe extern "C" fn cblas_scopy(n: blasint, x: *const f32, incx: blasint, y: *mut f32, incy: blasint) {
    let scopy = get_scopy();
    scopy(&n, x, &incx, y, &incy);
}

/// Double precision vector copy.
///
/// Copies vector x to vector y.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - dcopy must be registered via `register_dcopy`
#[no_mangle]
pub unsafe extern "C" fn cblas_dcopy(n: blasint, x: *const f64, incx: blasint, y: *mut f64, incy: blasint) {
    let dcopy = get_dcopy();
    dcopy(&n, x, &incx, y, &incy);
}

/// Single precision complex vector copy.
///
/// Copies vector x to vector y.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - ccopy must be registered via `register_ccopy`
#[no_mangle]
pub unsafe extern "C" fn cblas_ccopy(
    n: blasint,
    x: *const Complex32,
    incx: blasint,
    y: *mut Complex32,
    incy: blasint,
) {
    let ccopy = get_ccopy();
    ccopy(&n, x, &incx, y, &incy);
}

/// Double precision complex vector copy.
///
/// Copies vector x to vector y.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - zcopy must be registered via `register_zcopy`
#[no_mangle]
pub unsafe extern "C" fn cblas_zcopy(
    n: blasint,
    x: *const Complex64,
    incx: blasint,
    y: *mut Complex64,
    incy: blasint,
) {
    let zcopy = get_zcopy();
    zcopy(&n, x, &incx, y, &incy);
}

// =============================================================================
// Vector axpy (y = alpha*x + y)
// =============================================================================

/// Single precision axpy: y = alpha*x + y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - saxpy must be registered via `register_saxpy`
#[no_mangle]
pub unsafe extern "C" fn cblas_saxpy(
    n: blasint,
    alpha: f32,
    x: *const f32,
    incx: blasint,
    y: *mut f32,
    incy: blasint,
) {
    let saxpy = get_saxpy();
    saxpy(&n, &alpha, x, &incx, y, &incy);
}

/// Double precision axpy: y = alpha*x + y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - daxpy must be registered via `register_daxpy`
#[no_mangle]
pub unsafe extern "C" fn cblas_daxpy(
    n: blasint,
    alpha: f64,
    x: *const f64,
    incx: blasint,
    y: *mut f64,
    incy: blasint,
) {
    let daxpy = get_daxpy();
    daxpy(&n, &alpha, x, &incx, y, &incy);
}

/// Single precision complex axpy: y = alpha*x + y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - caxpy must be registered via `register_caxpy`
#[no_mangle]
pub unsafe extern "C" fn cblas_caxpy(
    n: blasint,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: blasint,
    y: *mut Complex32,
    incy: blasint,
) {
    let caxpy = get_caxpy();
    caxpy(&n, alpha, x, &incx, y, &incy);
}

/// Double precision complex axpy: y = alpha*x + y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - zaxpy must be registered via `register_zaxpy`
#[no_mangle]
pub unsafe extern "C" fn cblas_zaxpy(
    n: blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: blasint,
    y: *mut Complex64,
    incy: blasint,
) {
    let zaxpy = get_zaxpy();
    zaxpy(&n, alpha, x, &incx, y, &incy);
}

// =============================================================================
// Vector scale (x = alpha*x)
// =============================================================================

/// Single precision vector scaling: x = alpha*x
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - sscal must be registered via `register_sscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_sscal(n: blasint, alpha: f32, x: *mut f32, incx: blasint) {
    let sscal = get_sscal();
    sscal(&n, &alpha, x, &incx);
}

/// Double precision vector scaling: x = alpha*x
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - dscal must be registered via `register_dscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_dscal(n: blasint, alpha: f64, x: *mut f64, incx: blasint) {
    let dscal = get_dscal();
    dscal(&n, &alpha, x, &incx);
}

/// Single precision complex vector scaling: x = alpha*x
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - cscal must be registered via `register_cscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_cscal(n: blasint, alpha: *const Complex32, x: *mut Complex32, incx: blasint) {
    let cscal = get_cscal();
    cscal(&n, alpha, x, &incx);
}

/// Double precision complex vector scaling: x = alpha*x
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - zscal must be registered via `register_zscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_zscal(n: blasint, alpha: *const Complex64, x: *mut Complex64, incx: blasint) {
    let zscal = get_zscal();
    zscal(&n, alpha, x, &incx);
}

/// Scale complex vector by real scalar: x = alpha*x (single precision)
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - csscal must be registered via `register_csscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_csscal(n: blasint, alpha: f32, x: *mut Complex32, incx: blasint) {
    let csscal = get_csscal();
    csscal(&n, &alpha, x, &incx);
}

/// Scale complex vector by real scalar: x = alpha*x (double precision)
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - zdscal must be registered via `register_zdscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_zdscal(n: blasint, alpha: f64, x: *mut Complex64, incx: blasint) {
    let zdscal = get_zdscal();
    zdscal(&n, &alpha, x, &incx);
}
