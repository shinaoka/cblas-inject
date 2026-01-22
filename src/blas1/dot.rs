//! BLAS Level 1: Dot products, norms, and related functions.
//!
//! These are vector operations that do not involve matrix layout,
//! so no row-major conversion is needed - arguments are passed directly
//! to the Fortran BLAS functions.

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_cdotc_ptr, get_cdotu_ptr, get_complex_return_style, get_dasum, get_ddot, get_dnrm2,
    get_dsdot, get_dzasum, get_dznrm2, get_icamax, get_idamax, get_isamax, get_izamax, get_sasum,
    get_scasum, get_scnrm2, get_sdot, get_sdsdot, get_snrm2, get_zdotc_ptr, get_zdotu_ptr,
    CdotcFnPtr, CdotcHiddenFnPtr, CdotuFnPtr, CdotuHiddenFnPtr, ZdotcFnPtr, ZdotcHiddenFnPtr,
    ZdotuFnPtr, ZdotuHiddenFnPtr,
};
use crate::types::{blasint, ComplexReturnStyle};

// =============================================================================
// Dot products
// =============================================================================

/// Single precision dot product.
///
/// Computes: sum(x[i] * y[i])
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - sdot must be registered via `register_sdot`
#[no_mangle]
pub unsafe extern "C" fn cblas_sdot(
    n: blasint,
    x: *const f32,
    incx: blasint,
    y: *const f32,
    incy: blasint,
) -> f32 {
    let sdot = get_sdot();
    sdot(&n, x, &incx, y, &incy)
}

/// Double precision dot product.
///
/// Computes: sum(x[i] * y[i])
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - ddot must be registered via `register_ddot`
#[no_mangle]
pub unsafe extern "C" fn cblas_ddot(
    n: blasint,
    x: *const f64,
    incx: blasint,
    y: *const f64,
    incy: blasint,
) -> f64 {
    let ddot = get_ddot();
    ddot(&n, x, &incx, y, &incy)
}

/// Complex single precision dot product (unconjugated).
///
/// Computes: sum(x[i] * y[i])
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - cdotu must be registered via `register_cdotu`
#[no_mangle]
pub unsafe extern "C" fn cblas_cdotu_sub(
    n: blasint,
    x: *const Complex32,
    incx: blasint,
    y: *const Complex32,
    incy: blasint,
    dotu: *mut Complex32,
) {
    let ptr = get_cdotu_ptr();
    match get_complex_return_style() {
        ComplexReturnStyle::ReturnValue => {
            let f: CdotuFnPtr = std::mem::transmute(ptr);
            *dotu = f(&n, x, &incx, y, &incy);
        }
        ComplexReturnStyle::HiddenArgument => {
            let f: CdotuHiddenFnPtr = std::mem::transmute(ptr);
            f(dotu, &n, x, &incx, y, &incy);
        }
    }
}

/// Complex double precision dot product (unconjugated).
///
/// Computes: sum(x[i] * y[i])
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - zdotu must be registered via `register_zdotu`
#[no_mangle]
pub unsafe extern "C" fn cblas_zdotu_sub(
    n: blasint,
    x: *const Complex64,
    incx: blasint,
    y: *const Complex64,
    incy: blasint,
    dotu: *mut Complex64,
) {
    let ptr = get_zdotu_ptr();
    match get_complex_return_style() {
        ComplexReturnStyle::ReturnValue => {
            let f: ZdotuFnPtr = std::mem::transmute(ptr);
            *dotu = f(&n, x, &incx, y, &incy);
        }
        ComplexReturnStyle::HiddenArgument => {
            let f: ZdotuHiddenFnPtr = std::mem::transmute(ptr);
            f(dotu, &n, x, &incx, y, &incy);
        }
    }
}

/// Complex single precision dot product (conjugated).
///
/// Computes: sum(conj(x[i]) * y[i])
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - cdotc must be registered via `register_cdotc`
#[no_mangle]
pub unsafe extern "C" fn cblas_cdotc_sub(
    n: blasint,
    x: *const Complex32,
    incx: blasint,
    y: *const Complex32,
    incy: blasint,
    dotc: *mut Complex32,
) {
    let ptr = get_cdotc_ptr();
    match get_complex_return_style() {
        ComplexReturnStyle::ReturnValue => {
            let f: CdotcFnPtr = std::mem::transmute(ptr);
            *dotc = f(&n, x, &incx, y, &incy);
        }
        ComplexReturnStyle::HiddenArgument => {
            let f: CdotcHiddenFnPtr = std::mem::transmute(ptr);
            f(dotc, &n, x, &incx, y, &incy);
        }
    }
}

/// Complex double precision dot product (conjugated).
///
/// Computes: sum(conj(x[i]) * y[i])
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - zdotc must be registered via `register_zdotc`
#[no_mangle]
pub unsafe extern "C" fn cblas_zdotc_sub(
    n: blasint,
    x: *const Complex64,
    incx: blasint,
    y: *const Complex64,
    incy: blasint,
    dotc: *mut Complex64,
) {
    let ptr = get_zdotc_ptr();
    match get_complex_return_style() {
        ComplexReturnStyle::ReturnValue => {
            let f: ZdotcFnPtr = std::mem::transmute(ptr);
            *dotc = f(&n, x, &incx, y, &incy);
        }
        ComplexReturnStyle::HiddenArgument => {
            let f: ZdotcHiddenFnPtr = std::mem::transmute(ptr);
            f(dotc, &n, x, &incx, y, &incy);
        }
    }
}

// =============================================================================
// Extended precision dot products
// =============================================================================

/// Single precision dot product with double precision accumulation.
///
/// Computes: sb + sum(x[i] * y[i]) with double precision accumulation
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - sdsdot must be registered via `register_sdsdot`
#[no_mangle]
pub unsafe extern "C" fn cblas_sdsdot(
    n: blasint,
    sb: f32,
    x: *const f32,
    incx: blasint,
    y: *const f32,
    incy: blasint,
) -> f32 {
    let sdsdot = get_sdsdot();
    sdsdot(&n, &sb, x, &incx, y, &incy)
}

/// Double precision dot product of single precision vectors.
///
/// Computes: sum(x[i] * y[i]) with double precision result
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - dsdot must be registered via `register_dsdot`
#[no_mangle]
pub unsafe extern "C" fn cblas_dsdot(
    n: blasint,
    x: *const f32,
    incx: blasint,
    y: *const f32,
    incy: blasint,
) -> f64 {
    let dsdot = get_dsdot();
    dsdot(&n, x, &incx, y, &incy)
}

// =============================================================================
// Norms
// =============================================================================

/// Single precision Euclidean norm.
///
/// Computes: sqrt(sum(x[i]^2))
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - snrm2 must be registered via `register_snrm2`
#[no_mangle]
pub unsafe extern "C" fn cblas_snrm2(n: blasint, x: *const f32, incx: blasint) -> f32 {
    let snrm2 = get_snrm2();
    snrm2(&n, x, &incx)
}

/// Double precision Euclidean norm.
///
/// Computes: sqrt(sum(x[i]^2))
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - dnrm2 must be registered via `register_dnrm2`
#[no_mangle]
pub unsafe extern "C" fn cblas_dnrm2(n: blasint, x: *const f64, incx: blasint) -> f64 {
    let dnrm2 = get_dnrm2();
    dnrm2(&n, x, &incx)
}

/// Complex single precision Euclidean norm.
///
/// Computes: sqrt(sum(|x[i]|^2))
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - scnrm2 must be registered via `register_scnrm2`
#[no_mangle]
pub unsafe extern "C" fn cblas_scnrm2(n: blasint, x: *const Complex32, incx: blasint) -> f32 {
    let scnrm2 = get_scnrm2();
    scnrm2(&n, x, &incx)
}

/// Complex double precision Euclidean norm.
///
/// Computes: sqrt(sum(|x[i]|^2))
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - dznrm2 must be registered via `register_dznrm2`
#[no_mangle]
pub unsafe extern "C" fn cblas_dznrm2(n: blasint, x: *const Complex64, incx: blasint) -> f64 {
    let dznrm2 = get_dznrm2();
    dznrm2(&n, x, &incx)
}

// =============================================================================
// Sum of absolute values
// =============================================================================

/// Single precision sum of absolute values.
///
/// Computes: sum(|x[i]|)
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - sasum must be registered via `register_sasum`
#[no_mangle]
pub unsafe extern "C" fn cblas_sasum(n: blasint, x: *const f32, incx: blasint) -> f32 {
    let sasum = get_sasum();
    sasum(&n, x, &incx)
}

/// Double precision sum of absolute values.
///
/// Computes: sum(|x[i]|)
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - dasum must be registered via `register_dasum`
#[no_mangle]
pub unsafe extern "C" fn cblas_dasum(n: blasint, x: *const f64, incx: blasint) -> f64 {
    let dasum = get_dasum();
    dasum(&n, x, &incx)
}

/// Complex single precision sum of absolute values.
///
/// Computes: sum(|Re(x[i])| + |Im(x[i])|)
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - scasum must be registered via `register_scasum`
#[no_mangle]
pub unsafe extern "C" fn cblas_scasum(n: blasint, x: *const Complex32, incx: blasint) -> f32 {
    let scasum = get_scasum();
    scasum(&n, x, &incx)
}

/// Complex double precision sum of absolute values.
///
/// Computes: sum(|Re(x[i])| + |Im(x[i])|)
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - dzasum must be registered via `register_dzasum`
#[no_mangle]
pub unsafe extern "C" fn cblas_dzasum(n: blasint, x: *const Complex64, incx: blasint) -> f64 {
    let dzasum = get_dzasum();
    dzasum(&n, x, &incx)
}

// =============================================================================
// Index of maximum absolute value
// =============================================================================

/// Index of maximum absolute value (single precision).
///
/// Returns the index of the first element with maximum |x[i]|.
/// Note: CBLAS uses 0-based indexing, but Fortran BLAS returns 1-based index,
/// so we subtract 1 from the result.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - isamax must be registered via `register_isamax`
#[no_mangle]
pub unsafe extern "C" fn cblas_isamax(n: blasint, x: *const f32, incx: blasint) -> blasint {
    let isamax = get_isamax();
    let idx = isamax(&n, x, &incx);
    // Fortran returns 1-based index, convert to 0-based for CBLAS
    if idx > 0 {
        idx - 1
    } else {
        0
    }
}

/// Index of maximum absolute value (double precision).
///
/// Returns the index of the first element with maximum |x[i]|.
/// Note: CBLAS uses 0-based indexing, but Fortran BLAS returns 1-based index,
/// so we subtract 1 from the result.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - idamax must be registered via `register_idamax`
#[no_mangle]
pub unsafe extern "C" fn cblas_idamax(n: blasint, x: *const f64, incx: blasint) -> blasint {
    let idamax = get_idamax();
    let idx = idamax(&n, x, &incx);
    // Fortran returns 1-based index, convert to 0-based for CBLAS
    if idx > 0 {
        idx - 1
    } else {
        0
    }
}

/// Index of maximum absolute value (complex single precision).
///
/// Returns the index of the first element with maximum |Re(x[i])| + |Im(x[i])|.
/// Note: CBLAS uses 0-based indexing, but Fortran BLAS returns 1-based index,
/// so we subtract 1 from the result.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - icamax must be registered via `register_icamax`
#[no_mangle]
pub unsafe extern "C" fn cblas_icamax(n: blasint, x: *const Complex32, incx: blasint) -> blasint {
    let icamax = get_icamax();
    let idx = icamax(&n, x, &incx);
    // Fortran returns 1-based index, convert to 0-based for CBLAS
    if idx > 0 {
        idx - 1
    } else {
        0
    }
}

/// Index of maximum absolute value (complex double precision).
///
/// Returns the index of the first element with maximum |Re(x[i])| + |Im(x[i])|.
/// Note: CBLAS uses 0-based indexing, but Fortran BLAS returns 1-based index,
/// so we subtract 1 from the result.
///
/// # Safety
///
/// - All pointers must be valid and properly aligned
/// - Vector dimensions and increments must be consistent
/// - izamax must be registered via `register_izamax`
#[no_mangle]
pub unsafe extern "C" fn cblas_izamax(n: blasint, x: *const Complex64, incx: blasint) -> blasint {
    let izamax = get_izamax();
    let idx = izamax(&n, x, &incx);
    // Fortran returns 1-based index, convert to 0-based for CBLAS
    if idx > 0 {
        idx - 1
    } else {
        0
    }
}
