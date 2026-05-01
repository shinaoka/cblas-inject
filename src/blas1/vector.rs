//! BLAS Level 1: Vector operations (swap, copy, axpy, scal).
//!
//! These functions operate on vectors and do not require row-major conversion.

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_caxpy_for_ilp64_cblas, get_caxpy_for_lp64_cblas, get_ccopy_for_ilp64_cblas,
    get_ccopy_for_lp64_cblas, get_cscal_for_ilp64_cblas, get_cscal_for_lp64_cblas,
    get_csscal_for_ilp64_cblas, get_csscal_for_lp64_cblas, get_cswap_for_ilp64_cblas,
    get_cswap_for_lp64_cblas, get_daxpy_for_ilp64_cblas, get_daxpy_for_lp64_cblas,
    get_dcopy_for_ilp64_cblas, get_dcopy_for_lp64_cblas, get_dscal_for_ilp64_cblas,
    get_dscal_for_lp64_cblas, get_dswap_for_ilp64_cblas, get_dswap_for_lp64_cblas,
    get_saxpy_for_ilp64_cblas, get_saxpy_for_lp64_cblas, get_scopy_for_ilp64_cblas,
    get_scopy_for_lp64_cblas, get_sscal_for_ilp64_cblas, get_sscal_for_lp64_cblas,
    get_sswap_for_ilp64_cblas, get_sswap_for_lp64_cblas, get_zaxpy_for_ilp64_cblas,
    get_zaxpy_for_lp64_cblas, get_zcopy_for_ilp64_cblas, get_zcopy_for_lp64_cblas,
    get_zdscal_for_ilp64_cblas, get_zdscal_for_lp64_cblas, get_zscal_for_ilp64_cblas,
    get_zscal_for_lp64_cblas, get_zswap_for_ilp64_cblas, get_zswap_for_lp64_cblas, CaxpyProvider,
    CcopyProvider, CscalProvider, CsscalProvider, CswapProvider, DaxpyProvider, DcopyProvider,
    DscalProvider, DswapProvider, SaxpyProvider, ScopyProvider, SscalProvider, SswapProvider,
    ZaxpyProvider, ZcopyProvider, ZdscalProvider, ZscalProvider, ZswapProvider,
};

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
pub unsafe extern "C" fn cblas_sswap(n: i32, x: *mut f32, incx: i32, y: *mut f32, incy: i32) {
    let p = get_sswap_for_lp64_cblas();
    match p {
        SswapProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        SswapProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Single precision vector swap with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_sswap_64(n: i64, x: *mut f32, incx: i64, y: *mut f32, incy: i64) {
    let p = get_sswap_for_ilp64_cblas();
    if matches!(p, SswapProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_sswap_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        SswapProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        SswapProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
pub unsafe extern "C" fn cblas_dswap(n: i32, x: *mut f64, incx: i32, y: *mut f64, incy: i32) {
    let p = get_dswap_for_lp64_cblas();
    match p {
        DswapProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        DswapProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Double precision vector swap with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_dswap_64(n: i64, x: *mut f64, incx: i64, y: *mut f64, incy: i64) {
    let p = get_dswap_for_ilp64_cblas();
    if matches!(p, DswapProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dswap_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DswapProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        DswapProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
    n: i32,
    x: *mut Complex32,
    incx: i32,
    y: *mut Complex32,
    incy: i32,
) {
    let p = get_cswap_for_lp64_cblas();
    match p {
        CswapProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        CswapProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Single precision complex vector swap with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_cswap_64(
    n: i64,
    x: *mut Complex32,
    incx: i64,
    y: *mut Complex32,
    incy: i64,
) {
    let p = get_cswap_for_ilp64_cblas();
    if matches!(p, CswapProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_cswap_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        CswapProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        CswapProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
    n: i32,
    x: *mut Complex64,
    incx: i32,
    y: *mut Complex64,
    incy: i32,
) {
    let p = get_zswap_for_lp64_cblas();
    match p {
        ZswapProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        ZswapProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Double precision complex vector swap with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_zswap_64(
    n: i64,
    x: *mut Complex64,
    incx: i64,
    y: *mut Complex64,
    incy: i64,
) {
    let p = get_zswap_for_ilp64_cblas();
    if matches!(p, ZswapProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zswap_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZswapProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        ZswapProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
pub unsafe extern "C" fn cblas_scopy(n: i32, x: *const f32, incx: i32, y: *mut f32, incy: i32) {
    let p = get_scopy_for_lp64_cblas();
    match p {
        ScopyProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        ScopyProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Single precision vector copy with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_scopy_64(n: i64, x: *const f32, incx: i64, y: *mut f32, incy: i64) {
    let p = get_scopy_for_ilp64_cblas();
    if matches!(p, ScopyProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_scopy_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ScopyProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        ScopyProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
pub unsafe extern "C" fn cblas_dcopy(n: i32, x: *const f64, incx: i32, y: *mut f64, incy: i32) {
    let p = get_dcopy_for_lp64_cblas();
    match p {
        DcopyProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        DcopyProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Double precision vector copy with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_dcopy_64(n: i64, x: *const f64, incx: i64, y: *mut f64, incy: i64) {
    let p = get_dcopy_for_ilp64_cblas();
    if matches!(p, DcopyProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dcopy_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DcopyProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        DcopyProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
    n: i32,
    x: *const Complex32,
    incx: i32,
    y: *mut Complex32,
    incy: i32,
) {
    let p = get_ccopy_for_lp64_cblas();
    match p {
        CcopyProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        CcopyProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Single precision complex vector copy with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_ccopy_64(
    n: i64,
    x: *const Complex32,
    incx: i64,
    y: *mut Complex32,
    incy: i64,
) {
    let p = get_ccopy_for_ilp64_cblas();
    if matches!(p, CcopyProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_ccopy_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        CcopyProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        CcopyProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
    n: i32,
    x: *const Complex64,
    incx: i32,
    y: *mut Complex64,
    incy: i32,
) {
    let p = get_zcopy_for_lp64_cblas();
    match p {
        ZcopyProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        ZcopyProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Double precision complex vector copy with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_zcopy_64(
    n: i64,
    x: *const Complex64,
    incx: i64,
    y: *mut Complex64,
    incy: i64,
) {
    let p = get_zcopy_for_ilp64_cblas();
    if matches!(p, ZcopyProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zcopy_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZcopyProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        ZcopyProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
    n: i32,
    alpha: f32,
    x: *const f32,
    incx: i32,
    y: *mut f32,
    incy: i32,
) {
    let p = get_saxpy_for_lp64_cblas();
    match p {
        SaxpyProvider::Lp64(f) => f(&n, &alpha, x, &incx, y, &incy),
        SaxpyProvider::Ilp64(f) => f(&(n as i64), &alpha, x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Single precision axpy with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_saxpy_64(
    n: i64,
    alpha: f32,
    x: *const f32,
    incx: i64,
    y: *mut f32,
    incy: i64,
) {
    let p = get_saxpy_for_ilp64_cblas();
    if matches!(p, SaxpyProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_saxpy_64\0",
            [(1, n), (4, incx), (6, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        SaxpyProvider::Ilp64(f) => f(&n, &alpha, x, &incx, y, &incy),
        SaxpyProvider::Lp64(f) => f(&(n as i32), &alpha, x, &(incx as i32), y, &(incy as i32)),
    }
}

/// Double precision axpy: y = alpha*x + y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - daxpy must be registered via `register_daxpy`
#[no_mangle]
pub unsafe extern "C" fn cblas_daxpy(
    n: i32,
    alpha: f64,
    x: *const f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
) {
    let p = get_daxpy_for_lp64_cblas();
    match p {
        DaxpyProvider::Lp64(f) => f(&n, &alpha, x, &incx, y, &incy),
        DaxpyProvider::Ilp64(f) => f(&(n as i64), &alpha, x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Double precision axpy with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_daxpy_64(
    n: i64,
    alpha: f64,
    x: *const f64,
    incx: i64,
    y: *mut f64,
    incy: i64,
) {
    let p = get_daxpy_for_ilp64_cblas();
    if matches!(p, DaxpyProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_daxpy_64\0",
            [(1, n), (4, incx), (6, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        DaxpyProvider::Ilp64(f) => f(&n, &alpha, x, &incx, y, &incy),
        DaxpyProvider::Lp64(f) => f(&(n as i32), &alpha, x, &(incx as i32), y, &(incy as i32)),
    }
}

/// Single precision complex axpy: y = alpha*x + y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - caxpy must be registered via `register_caxpy`
#[no_mangle]
pub unsafe extern "C" fn cblas_caxpy(
    n: i32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i32,
    y: *mut Complex32,
    incy: i32,
) {
    let p = get_caxpy_for_lp64_cblas();
    match p {
        CaxpyProvider::Lp64(f) => f(&n, alpha, x, &incx, y, &incy),
        CaxpyProvider::Ilp64(f) => f(&(n as i64), alpha, x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Single precision complex axpy with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_caxpy_64(
    n: i64,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: i64,
    y: *mut Complex32,
    incy: i64,
) {
    let p = get_caxpy_for_ilp64_cblas();
    if matches!(p, CaxpyProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_caxpy_64\0",
            [(1, n), (4, incx), (6, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        CaxpyProvider::Ilp64(f) => f(&n, alpha, x, &incx, y, &incy),
        CaxpyProvider::Lp64(f) => f(&(n as i32), alpha, x, &(incx as i32), y, &(incy as i32)),
    }
}

/// Double precision complex axpy: y = alpha*x + y
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - zaxpy must be registered via `register_zaxpy`
#[no_mangle]
pub unsafe extern "C" fn cblas_zaxpy(
    n: i32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i32,
    y: *mut Complex64,
    incy: i32,
) {
    let p = get_zaxpy_for_lp64_cblas();
    match p {
        ZaxpyProvider::Lp64(f) => f(&n, alpha, x, &incx, y, &incy),
        ZaxpyProvider::Ilp64(f) => f(&(n as i64), alpha, x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Double precision complex axpy with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_zaxpy_64(
    n: i64,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: i64,
    y: *mut Complex64,
    incy: i64,
) {
    let p = get_zaxpy_for_ilp64_cblas();
    if matches!(p, ZaxpyProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_zaxpy_64\0",
            [(1, n), (4, incx), (6, incy)],
        )
        .is_none()
    {
        return;
    }

    match p {
        ZaxpyProvider::Ilp64(f) => f(&n, alpha, x, &incx, y, &incy),
        ZaxpyProvider::Lp64(f) => f(&(n as i32), alpha, x, &(incx as i32), y, &(incy as i32)),
    }
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
pub unsafe extern "C" fn cblas_sscal(n: i32, alpha: f32, x: *mut f32, incx: i32) {
    let p = get_sscal_for_lp64_cblas();
    match p {
        SscalProvider::Lp64(f) => f(&n, &alpha, x, &incx),
        SscalProvider::Ilp64(f) => f(&(n as i64), &alpha, x, &(incx as i64)),
    }
}

/// Single precision vector scaling with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_sscal_64(n: i64, alpha: f32, x: *mut f32, incx: i64) {
    let p = get_sscal_for_ilp64_cblas();
    if matches!(p, SscalProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_sscal_64\0", [(1, n), (4, incx)]).is_none()
    {
        return;
    }

    match p {
        SscalProvider::Ilp64(f) => f(&n, &alpha, x, &incx),
        SscalProvider::Lp64(f) => f(&(n as i32), &alpha, x, &(incx as i32)),
    }
}

/// Double precision vector scaling: x = alpha*x
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - dscal must be registered via `register_dscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32) {
    let p = get_dscal_for_lp64_cblas();
    match p {
        DscalProvider::Lp64(f) => f(&n, &alpha, x, &incx),
        DscalProvider::Ilp64(f) => f(&(n as i64), &alpha, x, &(incx as i64)),
    }
}

/// Double precision vector scaling with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_dscal_64(n: i64, alpha: f64, x: *mut f64, incx: i64) {
    let p = get_dscal_for_ilp64_cblas();
    if matches!(p, DscalProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_dscal_64\0", [(1, n), (4, incx)]).is_none()
    {
        return;
    }

    match p {
        DscalProvider::Ilp64(f) => f(&n, &alpha, x, &incx),
        DscalProvider::Lp64(f) => f(&(n as i32), &alpha, x, &(incx as i32)),
    }
}

/// Single precision complex vector scaling: x = alpha*x
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - cscal must be registered via `register_cscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_cscal(
    n: i32,
    alpha: *const Complex32,
    x: *mut Complex32,
    incx: i32,
) {
    let p = get_cscal_for_lp64_cblas();
    match p {
        CscalProvider::Lp64(f) => f(&n, alpha, x, &incx),
        CscalProvider::Ilp64(f) => f(&(n as i64), alpha, x, &(incx as i64)),
    }
}

/// Single precision complex vector scaling with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_cscal_64(
    n: i64,
    alpha: *const Complex32,
    x: *mut Complex32,
    incx: i64,
) {
    let p = get_cscal_for_ilp64_cblas();
    if matches!(p, CscalProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_cscal_64\0", [(1, n), (4, incx)]).is_none()
    {
        return;
    }

    match p {
        CscalProvider::Ilp64(f) => f(&n, alpha, x, &incx),
        CscalProvider::Lp64(f) => f(&(n as i32), alpha, x, &(incx as i32)),
    }
}

/// Double precision complex vector scaling: x = alpha*x
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - zscal must be registered via `register_zscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_zscal(
    n: i32,
    alpha: *const Complex64,
    x: *mut Complex64,
    incx: i32,
) {
    let p = get_zscal_for_lp64_cblas();
    match p {
        ZscalProvider::Lp64(f) => f(&n, alpha, x, &incx),
        ZscalProvider::Ilp64(f) => f(&(n as i64), alpha, x, &(incx as i64)),
    }
}

/// Double precision complex vector scaling with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_zscal_64(
    n: i64,
    alpha: *const Complex64,
    x: *mut Complex64,
    incx: i64,
) {
    let p = get_zscal_for_ilp64_cblas();
    if matches!(p, ZscalProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_zscal_64\0", [(1, n), (4, incx)]).is_none()
    {
        return;
    }

    match p {
        ZscalProvider::Ilp64(f) => f(&n, alpha, x, &incx),
        ZscalProvider::Lp64(f) => f(&(n as i32), alpha, x, &(incx as i32)),
    }
}

/// Scale complex vector by real scalar: x = alpha*x (single precision)
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - csscal must be registered via `register_csscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_csscal(n: i32, alpha: f32, x: *mut Complex32, incx: i32) {
    let p = get_csscal_for_lp64_cblas();
    match p {
        CsscalProvider::Lp64(f) => f(&n, &alpha, x, &incx),
        CsscalProvider::Ilp64(f) => f(&(n as i64), &alpha, x, &(incx as i64)),
    }
}

/// Scale complex vector by real scalar with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_csscal_64(n: i64, alpha: f32, x: *mut Complex32, incx: i64) {
    let p = get_csscal_for_ilp64_cblas();
    if matches!(p, CsscalProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_csscal_64\0", [(1, n), (4, incx)])
            .is_none()
    {
        return;
    }

    match p {
        CsscalProvider::Ilp64(f) => f(&n, &alpha, x, &incx),
        CsscalProvider::Lp64(f) => f(&(n as i32), &alpha, x, &(incx as i32)),
    }
}

/// Scale complex vector by real scalar: x = alpha*x (double precision)
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - zdscal must be registered via `register_zdscal`
#[no_mangle]
pub unsafe extern "C" fn cblas_zdscal(n: i32, alpha: f64, x: *mut Complex64, incx: i32) {
    let p = get_zdscal_for_lp64_cblas();
    match p {
        ZdscalProvider::Lp64(f) => f(&n, &alpha, x, &incx),
        ZdscalProvider::Ilp64(f) => f(&(n as i64), &alpha, x, &(incx as i64)),
    }
}

/// Scale complex vector by real scalar with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_zdscal_64(n: i64, alpha: f64, x: *mut Complex64, incx: i64) {
    let p = get_zdscal_for_ilp64_cblas();
    if matches!(p, ZdscalProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_zdscal_64\0", [(1, n), (4, incx)])
            .is_none()
    {
        return;
    }

    match p {
        ZdscalProvider::Ilp64(f) => f(&n, &alpha, x, &incx),
        ZdscalProvider::Lp64(f) => f(&(n as i32), &alpha, x, &(incx as i32)),
    }
}
