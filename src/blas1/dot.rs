//! BLAS Level 1: Dot products, norms, and related functions.
//!
//! These are vector operations that do not involve matrix layout,
//! so no row-major conversion is needed - arguments are passed directly
//! to the Fortran BLAS functions.

use num_complex::{Complex32, Complex64};

use crate::backend::{
    get_cdotc_for_ilp64_cblas, get_cdotc_for_lp64_cblas, get_cdotu_for_ilp64_cblas,
    get_cdotu_for_lp64_cblas, get_complex_return_style, get_dasum_for_ilp64_cblas,
    get_dasum_for_lp64_cblas, get_ddot_for_ilp64_cblas, get_ddot_for_lp64_cblas,
    get_dnrm2_for_ilp64_cblas, get_dnrm2_for_lp64_cblas, get_dsdot_for_ilp64_cblas,
    get_dsdot_for_lp64_cblas, get_dzasum_for_ilp64_cblas, get_dzasum_for_lp64_cblas,
    get_dznrm2_for_ilp64_cblas, get_dznrm2_for_lp64_cblas, get_icamax_for_ilp64_cblas,
    get_icamax_for_lp64_cblas, get_idamax_for_ilp64_cblas, get_idamax_for_lp64_cblas,
    get_isamax_for_ilp64_cblas, get_isamax_for_lp64_cblas, get_izamax_for_ilp64_cblas,
    get_izamax_for_lp64_cblas, get_sasum_for_ilp64_cblas, get_sasum_for_lp64_cblas,
    get_scasum_for_ilp64_cblas, get_scasum_for_lp64_cblas, get_scnrm2_for_ilp64_cblas,
    get_scnrm2_for_lp64_cblas, get_sdot_for_ilp64_cblas, get_sdot_for_lp64_cblas,
    get_sdsdot_for_ilp64_cblas, get_sdsdot_for_lp64_cblas, get_snrm2_for_ilp64_cblas,
    get_snrm2_for_lp64_cblas, get_zdotc_for_ilp64_cblas, get_zdotc_for_lp64_cblas,
    get_zdotu_for_ilp64_cblas, get_zdotu_for_lp64_cblas, BlasInt32, BlasInt64,
    CdotcHiddenIlp64FnPtr, CdotcHiddenLp64FnPtr, CdotcIlp64FnPtr, CdotcLp64FnPtr, CdotcProvider,
    CdotuHiddenIlp64FnPtr, CdotuHiddenLp64FnPtr, CdotuIlp64FnPtr, CdotuLp64FnPtr, CdotuProvider,
    DasumProvider, DdotProvider, Dnrm2Provider, DsdotProvider, DzasumProvider, Dznrm2Provider,
    IcamaxProvider, IdamaxProvider, IsamaxProvider, IzamaxProvider, SasumProvider, ScasumProvider,
    Scnrm2Provider, SdotProvider, SdsdotProvider, Snrm2Provider, ZdotcHiddenIlp64FnPtr,
    ZdotcHiddenLp64FnPtr, ZdotcIlp64FnPtr, ZdotcLp64FnPtr, ZdotcProvider, ZdotuHiddenIlp64FnPtr,
    ZdotuHiddenLp64FnPtr, ZdotuIlp64FnPtr, ZdotuLp64FnPtr, ZdotuProvider,
};
use crate::types::ComplexReturnStyle;

#[inline]
fn complex_dot_to_lp64_i64(
    n: i64,
    incx: i64,
    incy: i64,
) -> Option<(BlasInt32, BlasInt32, BlasInt32)> {
    Some((
        BlasInt32::try_from(n).ok()?,
        BlasInt32::try_from(incx).ok()?,
        BlasInt32::try_from(incy).ok()?,
    ))
}

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
    n: i32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
) -> f32 {
    let p = get_sdot_for_lp64_cblas();
    match p {
        SdotProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        SdotProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Single precision dot product with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_sdot_64(
    n: i64,
    x: *const f32,
    incx: i64,
    y: *const f32,
    incy: i64,
) -> f32 {
    let p = get_sdot_for_ilp64_cblas();
    if matches!(p, SdotProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_sdot_64\0", [(1, n), (3, incx), (5, incy)])
            .is_none()
    {
        return 0.0;
    }

    match p {
        SdotProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        SdotProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
    n: i32,
    x: *const f64,
    incx: i32,
    y: *const f64,
    incy: i32,
) -> f64 {
    let p = get_ddot_for_lp64_cblas();
    match p {
        DdotProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        DdotProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Double precision dot product with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_ddot_64(
    n: i64,
    x: *const f64,
    incx: i64,
    y: *const f64,
    incy: i64,
) -> f64 {
    let p = get_ddot_for_ilp64_cblas();
    if matches!(p, DdotProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_ddot_64\0", [(1, n), (3, incx), (5, incy)])
            .is_none()
    {
        return 0.0;
    }

    match p {
        DdotProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        DdotProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
    n: i32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    dotu: *mut Complex32,
) {
    match get_cdotu_for_lp64_cblas() {
        CdotuProvider::Lp64(ptr) => match get_complex_return_style() {
            ComplexReturnStyle::ReturnValue => {
                let f: CdotuLp64FnPtr = std::mem::transmute(ptr);
                *dotu = f(&n, x, &incx, y, &incy);
            }
            ComplexReturnStyle::HiddenArgument => {
                let f: CdotuHiddenLp64FnPtr = std::mem::transmute(ptr);
                f(dotu, &n, x, &incx, y, &incy);
            }
        },
        CdotuProvider::Ilp64(ptr) => {
            let n = BlasInt64::from(n);
            let incx = BlasInt64::from(incx);
            let incy = BlasInt64::from(incy);
            match get_complex_return_style() {
                ComplexReturnStyle::ReturnValue => {
                    let f: CdotuIlp64FnPtr = std::mem::transmute(ptr);
                    *dotu = f(&n, x, &incx, y, &incy);
                }
                ComplexReturnStyle::HiddenArgument => {
                    let f: CdotuHiddenIlp64FnPtr = std::mem::transmute(ptr);
                    f(dotu, &n, x, &incx, y, &incy);
                }
            }
        }
    };
}

/// Complex single precision dot product (unconjugated) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_cdotu_sub_64(
    n: i64,
    x: *const Complex32,
    incx: i64,
    y: *const Complex32,
    incy: i64,
    dotu: *mut Complex32,
) {
    match get_cdotu_for_ilp64_cblas() {
        CdotuProvider::Ilp64(ptr) => match get_complex_return_style() {
            ComplexReturnStyle::ReturnValue => {
                let f: CdotuIlp64FnPtr = std::mem::transmute(ptr);
                *dotu = f(&n, x, &incx, y, &incy);
            }
            ComplexReturnStyle::HiddenArgument => {
                let f: CdotuHiddenIlp64FnPtr = std::mem::transmute(ptr);
                f(dotu, &n, x, &incx, y, &incy);
            }
        },
        CdotuProvider::Lp64(ptr) => {
            let Some((n, incx, incy)) = complex_dot_to_lp64_i64(n, incx, incy) else {
                return;
            };
            match get_complex_return_style() {
                ComplexReturnStyle::ReturnValue => {
                    let f: CdotuLp64FnPtr = std::mem::transmute(ptr);
                    *dotu = f(&n, x, &incx, y, &incy);
                }
                ComplexReturnStyle::HiddenArgument => {
                    let f: CdotuHiddenLp64FnPtr = std::mem::transmute(ptr);
                    f(dotu, &n, x, &incx, y, &incy);
                }
            }
        }
    };
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
    n: i32,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    dotu: *mut Complex64,
) {
    match get_zdotu_for_lp64_cblas() {
        ZdotuProvider::Lp64(ptr) => match get_complex_return_style() {
            ComplexReturnStyle::ReturnValue => {
                let f: ZdotuLp64FnPtr = std::mem::transmute(ptr);
                *dotu = f(&n, x, &incx, y, &incy);
            }
            ComplexReturnStyle::HiddenArgument => {
                let f: ZdotuHiddenLp64FnPtr = std::mem::transmute(ptr);
                f(dotu, &n, x, &incx, y, &incy);
            }
        },
        ZdotuProvider::Ilp64(ptr) => {
            let n = BlasInt64::from(n);
            let incx = BlasInt64::from(incx);
            let incy = BlasInt64::from(incy);
            match get_complex_return_style() {
                ComplexReturnStyle::ReturnValue => {
                    let f: ZdotuIlp64FnPtr = std::mem::transmute(ptr);
                    *dotu = f(&n, x, &incx, y, &incy);
                }
                ComplexReturnStyle::HiddenArgument => {
                    let f: ZdotuHiddenIlp64FnPtr = std::mem::transmute(ptr);
                    f(dotu, &n, x, &incx, y, &incy);
                }
            }
        }
    };
}

/// Complex double precision dot product (unconjugated) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_zdotu_sub_64(
    n: i64,
    x: *const Complex64,
    incx: i64,
    y: *const Complex64,
    incy: i64,
    dotu: *mut Complex64,
) {
    match get_zdotu_for_ilp64_cblas() {
        ZdotuProvider::Ilp64(ptr) => match get_complex_return_style() {
            ComplexReturnStyle::ReturnValue => {
                let f: ZdotuIlp64FnPtr = std::mem::transmute(ptr);
                *dotu = f(&n, x, &incx, y, &incy);
            }
            ComplexReturnStyle::HiddenArgument => {
                let f: ZdotuHiddenIlp64FnPtr = std::mem::transmute(ptr);
                f(dotu, &n, x, &incx, y, &incy);
            }
        },
        ZdotuProvider::Lp64(ptr) => {
            let Some((n, incx, incy)) = complex_dot_to_lp64_i64(n, incx, incy) else {
                return;
            };
            match get_complex_return_style() {
                ComplexReturnStyle::ReturnValue => {
                    let f: ZdotuLp64FnPtr = std::mem::transmute(ptr);
                    *dotu = f(&n, x, &incx, y, &incy);
                }
                ComplexReturnStyle::HiddenArgument => {
                    let f: ZdotuHiddenLp64FnPtr = std::mem::transmute(ptr);
                    f(dotu, &n, x, &incx, y, &incy);
                }
            }
        }
    };
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
    n: i32,
    x: *const Complex32,
    incx: i32,
    y: *const Complex32,
    incy: i32,
    dotc: *mut Complex32,
) {
    match get_cdotc_for_lp64_cblas() {
        CdotcProvider::Lp64(ptr) => match get_complex_return_style() {
            ComplexReturnStyle::ReturnValue => {
                let f: CdotcLp64FnPtr = std::mem::transmute(ptr);
                *dotc = f(&n, x, &incx, y, &incy);
            }
            ComplexReturnStyle::HiddenArgument => {
                let f: CdotcHiddenLp64FnPtr = std::mem::transmute(ptr);
                f(dotc, &n, x, &incx, y, &incy);
            }
        },
        CdotcProvider::Ilp64(ptr) => {
            let n = BlasInt64::from(n);
            let incx = BlasInt64::from(incx);
            let incy = BlasInt64::from(incy);
            match get_complex_return_style() {
                ComplexReturnStyle::ReturnValue => {
                    let f: CdotcIlp64FnPtr = std::mem::transmute(ptr);
                    *dotc = f(&n, x, &incx, y, &incy);
                }
                ComplexReturnStyle::HiddenArgument => {
                    let f: CdotcHiddenIlp64FnPtr = std::mem::transmute(ptr);
                    f(dotc, &n, x, &incx, y, &incy);
                }
            }
        }
    };
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
    n: i32,
    x: *const Complex64,
    incx: i32,
    y: *const Complex64,
    incy: i32,
    dotc: *mut Complex64,
) {
    match get_zdotc_for_lp64_cblas() {
        ZdotcProvider::Lp64(ptr) => match get_complex_return_style() {
            ComplexReturnStyle::ReturnValue => {
                let f: ZdotcLp64FnPtr = std::mem::transmute(ptr);
                *dotc = f(&n, x, &incx, y, &incy);
            }
            ComplexReturnStyle::HiddenArgument => {
                let f: ZdotcHiddenLp64FnPtr = std::mem::transmute(ptr);
                f(dotc, &n, x, &incx, y, &incy);
            }
        },
        ZdotcProvider::Ilp64(ptr) => {
            let n = BlasInt64::from(n);
            let incx = BlasInt64::from(incx);
            let incy = BlasInt64::from(incy);
            match get_complex_return_style() {
                ComplexReturnStyle::ReturnValue => {
                    let f: ZdotcIlp64FnPtr = std::mem::transmute(ptr);
                    *dotc = f(&n, x, &incx, y, &incy);
                }
                ComplexReturnStyle::HiddenArgument => {
                    let f: ZdotcHiddenIlp64FnPtr = std::mem::transmute(ptr);
                    f(dotc, &n, x, &incx, y, &incy);
                }
            }
        }
    };
}

/// Complex single precision dot product (conjugated) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_cdotc_sub_64(
    n: i64,
    x: *const Complex32,
    incx: i64,
    y: *const Complex32,
    incy: i64,
    dotc: *mut Complex32,
) {
    match get_cdotc_for_ilp64_cblas() {
        CdotcProvider::Ilp64(ptr) => match get_complex_return_style() {
            ComplexReturnStyle::ReturnValue => {
                let f: CdotcIlp64FnPtr = std::mem::transmute(ptr);
                *dotc = f(&n, x, &incx, y, &incy);
            }
            ComplexReturnStyle::HiddenArgument => {
                let f: CdotcHiddenIlp64FnPtr = std::mem::transmute(ptr);
                f(dotc, &n, x, &incx, y, &incy);
            }
        },
        CdotcProvider::Lp64(ptr) => {
            let Some((n, incx, incy)) = complex_dot_to_lp64_i64(n, incx, incy) else {
                return;
            };
            match get_complex_return_style() {
                ComplexReturnStyle::ReturnValue => {
                    let f: CdotcLp64FnPtr = std::mem::transmute(ptr);
                    *dotc = f(&n, x, &incx, y, &incy);
                }
                ComplexReturnStyle::HiddenArgument => {
                    let f: CdotcHiddenLp64FnPtr = std::mem::transmute(ptr);
                    f(dotc, &n, x, &incx, y, &incy);
                }
            }
        }
    };
}

/// Complex double precision dot product (conjugated) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_zdotc_sub_64(
    n: i64,
    x: *const Complex64,
    incx: i64,
    y: *const Complex64,
    incy: i64,
    dotc: *mut Complex64,
) {
    match get_zdotc_for_ilp64_cblas() {
        ZdotcProvider::Ilp64(ptr) => match get_complex_return_style() {
            ComplexReturnStyle::ReturnValue => {
                let f: ZdotcIlp64FnPtr = std::mem::transmute(ptr);
                *dotc = f(&n, x, &incx, y, &incy);
            }
            ComplexReturnStyle::HiddenArgument => {
                let f: ZdotcHiddenIlp64FnPtr = std::mem::transmute(ptr);
                f(dotc, &n, x, &incx, y, &incy);
            }
        },
        ZdotcProvider::Lp64(ptr) => {
            let Some((n, incx, incy)) = complex_dot_to_lp64_i64(n, incx, incy) else {
                return;
            };
            match get_complex_return_style() {
                ComplexReturnStyle::ReturnValue => {
                    let f: ZdotcLp64FnPtr = std::mem::transmute(ptr);
                    *dotc = f(&n, x, &incx, y, &incy);
                }
                ComplexReturnStyle::HiddenArgument => {
                    let f: ZdotcHiddenLp64FnPtr = std::mem::transmute(ptr);
                    f(dotc, &n, x, &incx, y, &incy);
                }
            }
        }
    };
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
    n: i32,
    sb: f32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
) -> f32 {
    let p = get_sdsdot_for_lp64_cblas();
    match p {
        SdsdotProvider::Lp64(f) => f(&n, &sb, x, &incx, y, &incy),
        SdsdotProvider::Ilp64(f) => f(&(n as i64), &sb, x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Single precision dot product with double precision accumulation (ILP64 ABI).
#[no_mangle]
pub unsafe extern "C" fn cblas_sdsdot_64(
    n: i64,
    sb: f32,
    x: *const f32,
    incx: i64,
    y: *const f32,
    incy: i64,
) -> f32 {
    let p = get_sdsdot_for_ilp64_cblas();
    if matches!(p, SdsdotProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_sdsdot_64\0",
            [(1, n), (4, incx), (6, incy)],
        )
        .is_none()
    {
        return 0.0;
    }

    match p {
        SdsdotProvider::Ilp64(f) => f(&n, &sb, x, &incx, y, &incy),
        SdsdotProvider::Lp64(f) => f(&(n as i32), &sb, x, &(incx as i32), y, &(incy as i32)),
    }
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
    n: i32,
    x: *const f32,
    incx: i32,
    y: *const f32,
    incy: i32,
) -> f64 {
    let p = get_dsdot_for_lp64_cblas();
    match p {
        DsdotProvider::Lp64(f) => f(&n, x, &incx, y, &incy),
        DsdotProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64), y, &(incy as i64)),
    }
}

/// Double precision dot product of single precision vectors (ILP64 ABI).
#[no_mangle]
pub unsafe extern "C" fn cblas_dsdot_64(
    n: i64,
    x: *const f32,
    incx: i64,
    y: *const f32,
    incy: i64,
) -> f64 {
    let p = get_dsdot_for_ilp64_cblas();
    if matches!(p, DsdotProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(
            b"cblas_dsdot_64\0",
            [(1, n), (3, incx), (5, incy)],
        )
        .is_none()
    {
        return 0.0;
    }

    match p {
        DsdotProvider::Ilp64(f) => f(&n, x, &incx, y, &incy),
        DsdotProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32), y, &(incy as i32)),
    }
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
pub unsafe extern "C" fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32 {
    let p = get_snrm2_for_lp64_cblas();
    match p {
        Snrm2Provider::Lp64(f) => f(&n, x, &incx),
        Snrm2Provider::Ilp64(f) => f(&(n as i64), x, &(incx as i64)),
    }
}

/// Single precision Euclidean norm with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_snrm2_64(n: i64, x: *const f32, incx: i64) -> f32 {
    let p = get_snrm2_for_ilp64_cblas();
    if matches!(p, Snrm2Provider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_snrm2_64\0", [(1, n), (3, incx)]).is_none()
    {
        return 0.0;
    }

    match p {
        Snrm2Provider::Ilp64(f) => f(&n, x, &incx),
        Snrm2Provider::Lp64(f) => f(&(n as i32), x, &(incx as i32)),
    }
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
pub unsafe extern "C" fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64 {
    let p = get_dnrm2_for_lp64_cblas();
    match p {
        Dnrm2Provider::Lp64(f) => f(&n, x, &incx),
        Dnrm2Provider::Ilp64(f) => f(&(n as i64), x, &(incx as i64)),
    }
}

/// Double precision Euclidean norm with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_dnrm2_64(n: i64, x: *const f64, incx: i64) -> f64 {
    let p = get_dnrm2_for_ilp64_cblas();
    if matches!(p, Dnrm2Provider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_dnrm2_64\0", [(1, n), (3, incx)]).is_none()
    {
        return 0.0;
    }

    match p {
        Dnrm2Provider::Ilp64(f) => f(&n, x, &incx),
        Dnrm2Provider::Lp64(f) => f(&(n as i32), x, &(incx as i32)),
    }
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
pub unsafe extern "C" fn cblas_scnrm2(n: i32, x: *const Complex32, incx: i32) -> f32 {
    let p = get_scnrm2_for_lp64_cblas();
    match p {
        Scnrm2Provider::Lp64(f) => f(&n, x, &incx),
        Scnrm2Provider::Ilp64(f) => f(&(n as i64), x, &(incx as i64)),
    }
}

/// Complex single precision Euclidean norm with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_scnrm2_64(n: i64, x: *const Complex32, incx: i64) -> f32 {
    let p = get_scnrm2_for_ilp64_cblas();
    if matches!(p, Scnrm2Provider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_scnrm2_64\0", [(1, n), (3, incx)])
            .is_none()
    {
        return 0.0;
    }

    match p {
        Scnrm2Provider::Ilp64(f) => f(&n, x, &incx),
        Scnrm2Provider::Lp64(f) => f(&(n as i32), x, &(incx as i32)),
    }
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
pub unsafe extern "C" fn cblas_dznrm2(n: i32, x: *const Complex64, incx: i32) -> f64 {
    let p = get_dznrm2_for_lp64_cblas();
    match p {
        Dznrm2Provider::Lp64(f) => f(&n, x, &incx),
        Dznrm2Provider::Ilp64(f) => f(&(n as i64), x, &(incx as i64)),
    }
}

/// Complex double precision Euclidean norm with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_dznrm2_64(n: i64, x: *const Complex64, incx: i64) -> f64 {
    let p = get_dznrm2_for_ilp64_cblas();
    if matches!(p, Dznrm2Provider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_dznrm2_64\0", [(1, n), (3, incx)])
            .is_none()
    {
        return 0.0;
    }

    match p {
        Dznrm2Provider::Ilp64(f) => f(&n, x, &incx),
        Dznrm2Provider::Lp64(f) => f(&(n as i32), x, &(incx as i32)),
    }
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
pub unsafe extern "C" fn cblas_sasum(n: i32, x: *const f32, incx: i32) -> f32 {
    let p = get_sasum_for_lp64_cblas();
    match p {
        SasumProvider::Lp64(f) => f(&n, x, &incx),
        SasumProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64)),
    }
}

/// Single precision sum of absolute values with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_sasum_64(n: i64, x: *const f32, incx: i64) -> f32 {
    let p = get_sasum_for_ilp64_cblas();
    if matches!(p, SasumProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_sasum_64\0", [(1, n), (3, incx)]).is_none()
    {
        return 0.0;
    }

    match p {
        SasumProvider::Ilp64(f) => f(&n, x, &incx),
        SasumProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32)),
    }
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
pub unsafe extern "C" fn cblas_dasum(n: i32, x: *const f64, incx: i32) -> f64 {
    let p = get_dasum_for_lp64_cblas();
    match p {
        DasumProvider::Lp64(f) => f(&n, x, &incx),
        DasumProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64)),
    }
}

/// Double precision sum of absolute values with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_dasum_64(n: i64, x: *const f64, incx: i64) -> f64 {
    let p = get_dasum_for_ilp64_cblas();
    if matches!(p, DasumProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_dasum_64\0", [(1, n), (3, incx)]).is_none()
    {
        return 0.0;
    }

    match p {
        DasumProvider::Ilp64(f) => f(&n, x, &incx),
        DasumProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32)),
    }
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
pub unsafe extern "C" fn cblas_scasum(n: i32, x: *const Complex32, incx: i32) -> f32 {
    let p = get_scasum_for_lp64_cblas();
    match p {
        ScasumProvider::Lp64(f) => f(&n, x, &incx),
        ScasumProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64)),
    }
}

/// Complex single precision sum of absolute values with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_scasum_64(n: i64, x: *const Complex32, incx: i64) -> f32 {
    let p = get_scasum_for_ilp64_cblas();
    if matches!(p, ScasumProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_scasum_64\0", [(1, n), (3, incx)])
            .is_none()
    {
        return 0.0;
    }

    match p {
        ScasumProvider::Ilp64(f) => f(&n, x, &incx),
        ScasumProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32)),
    }
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
pub unsafe extern "C" fn cblas_dzasum(n: i32, x: *const Complex64, incx: i32) -> f64 {
    let p = get_dzasum_for_lp64_cblas();
    match p {
        DzasumProvider::Lp64(f) => f(&n, x, &incx),
        DzasumProvider::Ilp64(f) => f(&(n as i64), x, &(incx as i64)),
    }
}

/// Complex double precision sum of absolute values with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_dzasum_64(n: i64, x: *const Complex64, incx: i64) -> f64 {
    let p = get_dzasum_for_ilp64_cblas();
    if matches!(p, DzasumProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_dzasum_64\0", [(1, n), (3, incx)])
            .is_none()
    {
        return 0.0;
    }

    match p {
        DzasumProvider::Ilp64(f) => f(&n, x, &incx),
        DzasumProvider::Lp64(f) => f(&(n as i32), x, &(incx as i32)),
    }
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
pub unsafe extern "C" fn cblas_isamax(n: i32, x: *const f32, incx: i32) -> i32 {
    let p = get_isamax_for_lp64_cblas();
    match p {
        IsamaxProvider::Lp64(f) => {
            let idx = f(&n, x, &incx);
            if idx > 0 {
                idx - 1
            } else {
                0
            }
        }
        IsamaxProvider::Ilp64(f) => {
            let idx = f(&(n as i64), x, &(incx as i64));
            if idx > 0 {
                (idx - 1) as i32
            } else {
                0
            }
        }
    }
}

/// Index of maximum absolute value (single precision) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_isamax_64(n: i64, x: *const f32, incx: i64) -> i64 {
    let p = get_isamax_for_ilp64_cblas();
    if matches!(p, IsamaxProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_isamax_64\0", [(1, n), (3, incx)])
            .is_none()
    {
        return 0;
    }

    match p {
        IsamaxProvider::Ilp64(f) => {
            let idx = f(&n, x, &incx);
            if idx > 0 {
                idx - 1
            } else {
                0
            }
        }
        IsamaxProvider::Lp64(f) => {
            let idx = f(&(n as i32), x, &(incx as i32));
            if idx > 0 {
                (idx - 1) as i64
            } else {
                0
            }
        }
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
pub unsafe extern "C" fn cblas_idamax(n: i32, x: *const f64, incx: i32) -> i32 {
    let p = get_idamax_for_lp64_cblas();
    match p {
        IdamaxProvider::Lp64(f) => {
            let idx = f(&n, x, &incx);
            if idx > 0 {
                idx - 1
            } else {
                0
            }
        }
        IdamaxProvider::Ilp64(f) => {
            let idx = f(&(n as i64), x, &(incx as i64));
            if idx > 0 {
                (idx - 1) as i32
            } else {
                0
            }
        }
    }
}

/// Index of maximum absolute value (double precision) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_idamax_64(n: i64, x: *const f64, incx: i64) -> i64 {
    let p = get_idamax_for_ilp64_cblas();
    if matches!(p, IdamaxProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_idamax_64\0", [(1, n), (3, incx)])
            .is_none()
    {
        return 0;
    }

    match p {
        IdamaxProvider::Ilp64(f) => {
            let idx = f(&n, x, &incx);
            if idx > 0 {
                idx - 1
            } else {
                0
            }
        }
        IdamaxProvider::Lp64(f) => {
            let idx = f(&(n as i32), x, &(incx as i32));
            if idx > 0 {
                (idx - 1) as i64
            } else {
                0
            }
        }
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
pub unsafe extern "C" fn cblas_icamax(n: i32, x: *const Complex32, incx: i32) -> i32 {
    let p = get_icamax_for_lp64_cblas();
    match p {
        IcamaxProvider::Lp64(f) => {
            let idx = f(&n, x, &incx);
            if idx > 0 {
                idx - 1
            } else {
                0
            }
        }
        IcamaxProvider::Ilp64(f) => {
            let idx = f(&(n as i64), x, &(incx as i64));
            if idx > 0 {
                (idx - 1) as i32
            } else {
                0
            }
        }
    }
}

/// Index of maximum absolute value (complex single precision) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_icamax_64(n: i64, x: *const Complex32, incx: i64) -> i64 {
    let p = get_icamax_for_ilp64_cblas();
    if matches!(p, IcamaxProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_icamax_64\0", [(1, n), (3, incx)])
            .is_none()
    {
        return 0;
    }

    match p {
        IcamaxProvider::Ilp64(f) => {
            let idx = f(&n, x, &incx);
            if idx > 0 {
                idx - 1
            } else {
                0
            }
        }
        IcamaxProvider::Lp64(f) => {
            let idx = f(&(n as i32), x, &(incx as i32));
            if idx > 0 {
                (idx - 1) as i64
            } else {
                0
            }
        }
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
pub unsafe extern "C" fn cblas_izamax(n: i32, x: *const Complex64, incx: i32) -> i32 {
    let p = get_izamax_for_lp64_cblas();
    match p {
        IzamaxProvider::Lp64(f) => {
            let idx = f(&n, x, &incx);
            if idx > 0 {
                idx - 1
            } else {
                0
            }
        }
        IzamaxProvider::Ilp64(f) => {
            let idx = f(&(n as i64), x, &(incx as i64));
            if idx > 0 {
                (idx - 1) as i32
            } else {
                0
            }
        }
    }
}

/// Index of maximum absolute value (complex double precision) with ILP64 integer ABI.
#[no_mangle]
pub unsafe extern "C" fn cblas_izamax_64(n: i64, x: *const Complex64, incx: i64) -> i64 {
    let p = get_izamax_for_ilp64_cblas();
    if matches!(p, IzamaxProvider::Lp64(_))
        && crate::int_convert::to_lp64_array_i64(b"cblas_izamax_64\0", [(1, n), (3, incx)])
            .is_none()
    {
        return 0;
    }

    match p {
        IzamaxProvider::Ilp64(f) => {
            let idx = f(&n, x, &incx);
            if idx > 0 {
                idx - 1
            } else {
                0
            }
        }
        IzamaxProvider::Lp64(f) => {
            let idx = f(&(n as i32), x, &(incx as i32));
            if idx > 0 {
                (idx - 1) as i64
            } else {
                0
            }
        }
    }
}
