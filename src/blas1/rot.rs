//! BLAS Level 1: Givens rotations and auxiliary functions.

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_dcabs1, get_drot_for_ilp64_cblas, get_drot_for_lp64_cblas, get_drotg,
    get_drotm_for_ilp64_cblas, get_drotm_for_lp64_cblas, get_drotmg, get_scabs1,
    get_srot_for_ilp64_cblas, get_srot_for_lp64_cblas, get_srotg, get_srotm_for_ilp64_cblas,
    get_srotm_for_lp64_cblas, get_srotmg, DrotProvider, DrotmProvider, SrotProvider, SrotmProvider,
};

/// Apply Givens rotation (double precision).
///
/// Applies the rotation:
/// ```text
/// x[i] = c*x[i] + s*y[i]
/// y[i] = -s*x[i] + c*y[i]
/// ```
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Arrays must have at least `n` elements with the given stride
/// - drot must be registered via `register_drot`
#[no_mangle]
pub unsafe extern "C" fn cblas_drot(
    n: i32,
    x: *mut f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
    c: f64,
    s: f64,
) {
    let p = get_drot_for_lp64_cblas();
    match p {
        DrotProvider::Lp64(f) => f(&n, x, &incx, y, &incy, &c, &s),
        DrotProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64), &c, &s),
    }
}

/// Apply Givens rotation (double precision) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_drot_64(
    n: i64,
    x: *mut f64,
    incx: i64,
    y: *mut f64,
    incy: i64,
    c: f64,
    s: f64,
) {
    let p = get_drot_for_ilp64_cblas();
    if matches!(p, DrotProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_drot_64\0", [(1, n), (3, incx), (5, incy)])
            .is_none()
    {
        return;
    }

    match p {
        DrotProvider::Ilp64(f) => f(&n, x, &incx, y, &incy, &c, &s),
        DrotProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32), &c, &s),
    }
}

/// Apply Givens rotation (single precision).
///
/// Applies the rotation:
/// ```text
/// x[i] = c*x[i] + s*y[i]
/// y[i] = -s*x[i] + c*y[i]
/// ```
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Arrays must have at least `n` elements with the given stride
/// - srot must be registered via `register_srot`
#[no_mangle]
pub unsafe extern "C" fn cblas_srot(
    n: i32,
    x: *mut f32,
    incx: i32,
    y: *mut f32,
    incy: i32,
    c: f32,
    s: f32,
) {
    let p = get_srot_for_lp64_cblas();
    match p {
        SrotProvider::Lp64(f) => f(&n, x, &incx, y, &incy, &c, &s),
        SrotProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64), &c, &s),
    }
}

/// Apply Givens rotation (single precision) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_srot_64(
    n: i64,
    x: *mut f32,
    incx: i64,
    y: *mut f32,
    incy: i64,
    c: f32,
    s: f32,
) {
    let p = get_srot_for_ilp64_cblas();
    if matches!(p, SrotProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_srot_64\0", [(1, n), (3, incx), (5, incy)])
            .is_none()
    {
        return;
    }

    match p {
        SrotProvider::Ilp64(f) => f(&n, x, &incx, y, &incy, &c, &s),
        SrotProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32), &c, &s),
    }
}

/// Generate Givens rotation (double precision).
///
/// Computes `c` and `s` such that:
/// ```text
/// [ c  s ] [ a ]   [ r ]
/// [-s  c ] [ b ] = [ 0 ]
/// ```
///
/// The values of `a` and `b` are overwritten with `r` and `z` respectively,
/// where `z` encodes the rotation for reconstruction.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - drotg must be registered via `register_drotg`
#[no_mangle]
pub unsafe extern "C" fn cblas_drotg(a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64) {
    let drotg = get_drotg();
    drotg(a, b, c, s);
}

/// Generate Givens rotation (single precision).
///
/// Computes `c` and `s` such that:
/// ```text
/// [ c  s ] [ a ]   [ r ]
/// [-s  c ] [ b ] = [ 0 ]
/// ```
///
/// The values of `a` and `b` are overwritten with `r` and `z` respectively,
/// where `z` encodes the rotation for reconstruction.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - srotg must be registered via `register_srotg`
#[no_mangle]
pub unsafe extern "C" fn cblas_srotg(a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32) {
    let srotg = get_srotg();
    srotg(a, b, c, s);
}

/// Apply modified Givens rotation (double precision).
///
/// Applies the modified Givens rotation specified by the 5-element parameter array `p`:
/// - `p[0]` = flag indicating the form of the rotation matrix
/// - `p[1..5]` = rotation matrix elements (interpretation depends on flag)
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Arrays must have at least `n` elements with the given stride
/// - Parameter array `p` must have at least 5 elements
/// - drotm must be registered via `register_drotm`
#[no_mangle]
pub unsafe extern "C" fn cblas_drotm(
    n: i32,
    x: *mut f64,
    incx: i32,
    y: *mut f64,
    incy: i32,
    p: *const f64,
) {
    let pv = get_drotm_for_lp64_cblas();
    match pv {
        DrotmProvider::Lp64(f) => f(&n, x, &incx, y, &incy, p),
        DrotmProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64), p),
    }
}

/// Apply modified Givens rotation (double precision) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_drotm_64(
    n: i64,
    x: *mut f64,
    incx: i64,
    y: *mut f64,
    incy: i64,
    p: *const f64,
) {
    let pv = get_drotm_for_ilp64_cblas();
    if matches!(pv, DrotmProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_drotm_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return;
    }

    match pv {
        DrotmProvider::Ilp64(f) => f(&n, x, &incx, y, &incy, p),
        DrotmProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32), p),
    }
}

/// Apply modified Givens rotation (single precision).
///
/// Applies the modified Givens rotation specified by the 5-element parameter array `p`:
/// - `p[0]` = flag indicating the form of the rotation matrix
/// - `p[1..5]` = rotation matrix elements (interpretation depends on flag)
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Arrays must have at least `n` elements with the given stride
/// - Parameter array `p` must have at least 5 elements
/// - srotm must be registered via `register_srotm`
#[no_mangle]
pub unsafe extern "C" fn cblas_srotm(
    n: i32,
    x: *mut f32,
    incx: i32,
    y: *mut f32,
    incy: i32,
    p: *const f32,
) {
    let pv = get_srotm_for_lp64_cblas();
    match pv {
        SrotmProvider::Lp64(f) => f(&n, x, &incx, y, &incy, p),
        SrotmProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64), p),
    }
}

/// Apply modified Givens rotation (single precision) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_srotm_64(
    n: i64,
    x: *mut f32,
    incx: i64,
    y: *mut f32,
    incy: i64,
    p: *const f32,
) {
    let pv = get_srotm_for_ilp64_cblas();
    if matches!(pv, SrotmProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_srotm_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return;
    }

    match pv {
        SrotmProvider::Ilp64(f) => f(&n, x, &incx, y, &incy, p),
        SrotmProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32), p),
    }
}

/// Generate modified Givens rotation (double precision).
///
/// Constructs a modified Givens rotation that eliminates the second element
/// of a 2-element vector while scaling by diagonal matrices.
///
/// # Parameters
///
/// - `d1`, `d2`: Scaling factors (modified on output)
/// - `b1`: First element of input vector (modified on output)
/// - `b2`: Second element of input vector
/// - `p`: 5-element output array containing the rotation parameters
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Parameter array `p` must have at least 5 elements
/// - drotmg must be registered via `register_drotmg`
#[no_mangle]
pub unsafe extern "C" fn cblas_drotmg(
    d1: *mut f64,
    d2: *mut f64,
    b1: *mut f64,
    b2: f64,
    p: *mut f64,
) {
    let drotmg = get_drotmg();
    drotmg(d1, d2, b1, &b2, p);
}

/// Generate modified Givens rotation (single precision).
///
/// Constructs a modified Givens rotation that eliminates the second element
/// of a 2-element vector while scaling by diagonal matrices.
///
/// # Parameters
///
/// - `d1`, `d2`: Scaling factors (modified on output)
/// - `b1`: First element of input vector (modified on output)
/// - `b2`: Second element of input vector
/// - `p`: 5-element output array containing the rotation parameters
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Parameter array `p` must have at least 5 elements
/// - srotmg must be registered via `register_srotmg`
#[no_mangle]
pub unsafe extern "C" fn cblas_srotmg(
    d1: *mut f32,
    d2: *mut f32,
    b1: *mut f32,
    b2: f32,
    p: *mut f32,
) {
    let srotmg = get_srotmg();
    srotmg(d1, d2, b1, &b2, p);
}

/// Compute |Re(z)| + |Im(z)| for a complex number (double precision).
///
/// This is used in BLAS/LAPACK as a simple proxy for complex absolute value
/// that avoids square root computation.
///
/// # Safety
///
/// - The pointer must be valid and properly aligned
/// - dcabs1 must be registered via `register_dcabs1`
#[no_mangle]
pub unsafe extern "C" fn cblas_dcabs1(z: *const Complex64) -> f64 {
    let dcabs1 = get_dcabs1();
    dcabs1(z)
}

/// Compute |Re(z)| + |Im(z)| for a complex number (single precision).
///
/// This is used in BLAS/LAPACK as a simple proxy for complex absolute value
/// that avoids square root computation.
///
/// # Safety
///
/// - The pointer must be valid and properly aligned
/// - scabs1 must be registered via `register_scabs1`
#[no_mangle]
pub unsafe extern "C" fn cblas_scabs1(z: *const Complex32) -> f32 {
    let scabs1 = get_scabs1();
    scabs1(z)
}
