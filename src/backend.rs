//! Fortran BLAS/LAPACK function pointer registration.
//!
//! This module provides the infrastructure for registering Fortran BLAS/LAPACK
//! function pointers at runtime. Each function has its own `OnceLock` to allow
//! partial registration (only register the functions you need).

use std::ffi::{c_char, c_void};
use std::sync::{Mutex, MutexGuard, OnceLock};

use num_complex::{Complex32, Complex64};

use crate::blasint;

pub type BlasInt32 = i32;
pub type BlasInt64 = i64;

/// Macro to generate dual LP64/ILP64 backend infrastructure for one function.
macro_rules! define_dual_backend {
    ($name:ident, $name_str:literal, $lp64_type:ty, $ilp64_type:ty, $provider:ident) => {
        paste::paste! {
            #[allow(non_upper_case_globals)]
            static [<$name _LP64>]: std::sync::OnceLock<$lp64_type> = std::sync::OnceLock::new();
            #[allow(non_upper_case_globals)]
            static [<$name _ILP64>]: std::sync::OnceLock<$ilp64_type> = std::sync::OnceLock::new();
        }

        paste::paste! {
            #[no_mangle]
            pub unsafe extern "C" fn [<cblas_inject_register_ $name_str _lp64>](f: *const std::ffi::c_void) -> i32 {
                if f.is_null() { return crate::backend::CBLAS_INJECT_STATUS_NULL_POINTER; }
                let f: $lp64_type = unsafe { std::mem::transmute(f) };
                match paste::paste!([<$name _LP64>]).set(f) {
                    Ok(()) => crate::backend::CBLAS_INJECT_STATUS_OK,
                    Err(_) => crate::backend::CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
                }
            }

            #[no_mangle]
            pub unsafe extern "C" fn [<cblas_inject_register_ $name_str _ilp64>](f: *const std::ffi::c_void) -> i32 {
                if f.is_null() { return crate::backend::CBLAS_INJECT_STATUS_NULL_POINTER; }
                let f: $ilp64_type = unsafe { std::mem::transmute(f) };
                match paste::paste!([<$name _ILP64>]).set(f) {
                    Ok(()) => crate::backend::CBLAS_INJECT_STATUS_OK,
                    Err(_) => crate::backend::CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
                }
            }
        }

        #[derive(Clone, Copy)]
        pub(crate) enum $provider {
            Lp64($lp64_type),
            Ilp64($ilp64_type),
        }

        paste::paste! {
            pub(crate) fn [<get_ $name:lower _for_lp64_cblas>]() -> $provider {
                if let Some(f) = [<$name _LP64>].get() {
                    return $provider::Lp64(*f);
                }
                if let Some(f) = [<$name _ILP64>].get() {
                    return $provider::Ilp64(*f);
                }
                panic!("{} not registered: call cblas_inject_register_{}_lp64() or cblas_inject_register_{}_ilp64() first", stringify!($name), $name_str, $name_str);
            }

            pub(crate) fn [<get_ $name:lower _for_ilp64_cblas>]() -> $provider {
                if let Some(f) = [<$name _ILP64>].get() {
                    return $provider::Ilp64(*f);
                }
                if let Some(f) = [<$name _LP64>].get() {
                    return $provider::Lp64(*f);
                }
                panic!("{} not registered", stringify!($name));
            }
        }
    };
}

/// C API status code for a successful operation.
pub const CBLAS_INJECT_STATUS_OK: i32 = 0;
/// C API status code for a null function pointer.
pub const CBLAS_INJECT_STATUS_NULL_POINTER: i32 = 1;
/// C API status code for a function that has already been registered.
pub const CBLAS_INJECT_STATUS_ALREADY_REGISTERED: i32 = 2;

// =============================================================================
// Fortran BLAS function pointer types
// =============================================================================

// BLAS Level 1: Vector-Vector operations

/// Fortran sswap function pointer type (single precision vector swap)
pub type SswapFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut f32,
    incx: *const blasint,
    y: *mut f32,
    incy: *const blasint,
);

pub type SswapLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *mut f32,
    incx: *const BlasInt32,
    y: *mut f32,
    incy: *const BlasInt32,
);

pub type SswapIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *mut f32,
    incx: *const BlasInt64,
    y: *mut f32,
    incy: *const BlasInt64,
);


/// Fortran dswap function pointer type (double precision vector swap)
pub type DswapFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut f64,
    incx: *const blasint,
    y: *mut f64,
    incy: *const blasint,
);

pub type DswapLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *mut f64,
    incx: *const BlasInt32,
    y: *mut f64,
    incy: *const BlasInt32,
);

pub type DswapIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *mut f64,
    incx: *const BlasInt64,
    y: *mut f64,
    incy: *const BlasInt64,
);


/// Fortran cswap function pointer type (single precision complex vector swap)
pub type CswapFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut Complex32,
    incx: *const blasint,
    y: *mut Complex32,
    incy: *const blasint,
);

pub type CswapLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *mut Complex32,
    incx: *const BlasInt32,
    y: *mut Complex32,
    incy: *const BlasInt32,
);

pub type CswapIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *mut Complex32,
    incx: *const BlasInt64,
    y: *mut Complex32,
    incy: *const BlasInt64,
);


/// Fortran zswap function pointer type (double precision complex vector swap)
pub type ZswapFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut Complex64,
    incx: *const blasint,
    y: *mut Complex64,
    incy: *const blasint,
);

pub type ZswapLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *mut Complex64,
    incx: *const BlasInt32,
    y: *mut Complex64,
    incy: *const BlasInt32,
);

pub type ZswapIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *mut Complex64,
    incx: *const BlasInt64,
    y: *mut Complex64,
    incy: *const BlasInt64,
);


/// Fortran scopy function pointer type (single precision vector copy)
pub type ScopyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const f32,
    incx: *const blasint,
    y: *mut f32,
    incy: *const blasint,
);

pub type ScopyLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const f32,
    incx: *const BlasInt32,
    y: *mut f32,
    incy: *const BlasInt32,
);

pub type ScopyIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const f32,
    incx: *const BlasInt64,
    y: *mut f32,
    incy: *const BlasInt64,
);


/// Fortran dcopy function pointer type (double precision vector copy)
pub type DcopyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const f64,
    incx: *const blasint,
    y: *mut f64,
    incy: *const blasint,
);

pub type DcopyLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const f64,
    incx: *const BlasInt32,
    y: *mut f64,
    incy: *const BlasInt32,
);

pub type DcopyIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const f64,
    incx: *const BlasInt64,
    y: *mut f64,
    incy: *const BlasInt64,
);


/// Fortran ccopy function pointer type (single precision complex vector copy)
pub type CcopyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    y: *mut Complex32,
    incy: *const blasint,
);

pub type CcopyLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const Complex32,
    incx: *const BlasInt32,
    y: *mut Complex32,
    incy: *const BlasInt32,
);

pub type CcopyIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const Complex32,
    incx: *const BlasInt64,
    y: *mut Complex32,
    incy: *const BlasInt64,
);


/// Fortran zcopy function pointer type (double precision complex vector copy)
pub type ZcopyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    y: *mut Complex64,
    incy: *const blasint,
);

pub type ZcopyLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const Complex64,
    incx: *const BlasInt32,
    y: *mut Complex64,
    incy: *const BlasInt32,
);

pub type ZcopyIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const Complex64,
    incx: *const BlasInt64,
    y: *mut Complex64,
    incy: *const BlasInt64,
);


/// Fortran saxpy function pointer type (single precision y = alpha*x + y)
pub type SaxpyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const f32,
    x: *const f32,
    incx: *const blasint,
    y: *mut f32,
    incy: *const blasint,
);

pub type SaxpyLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt32,
    y: *mut f32,
    incy: *const BlasInt32,
);

pub type SaxpyIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt64,
    y: *mut f32,
    incy: *const BlasInt64,
);


/// Fortran daxpy function pointer type (double precision y = alpha*x + y)
pub type DaxpyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const f64,
    x: *const f64,
    incx: *const blasint,
    y: *mut f64,
    incy: *const blasint,
);

pub type DaxpyLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt32,
    y: *mut f64,
    incy: *const BlasInt32,
);

pub type DaxpyIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt64,
    y: *mut f64,
    incy: *const BlasInt64,
);


/// Fortran caxpy function pointer type (single precision complex y = alpha*x + y)
pub type CaxpyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const blasint,
    y: *mut Complex32,
    incy: *const blasint,
);

pub type CaxpyLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt32,
    y: *mut Complex32,
    incy: *const BlasInt32,
);

pub type CaxpyIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt64,
    y: *mut Complex32,
    incy: *const BlasInt64,
);


/// Fortran zaxpy function pointer type (double precision complex y = alpha*x + y)
pub type ZaxpyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const blasint,
    y: *mut Complex64,
    incy: *const blasint,
);

pub type ZaxpyLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt32,
    y: *mut Complex64,
    incy: *const BlasInt32,
);

pub type ZaxpyIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt64,
    y: *mut Complex64,
    incy: *const BlasInt64,
);


/// Fortran sscal function pointer type (single precision vector scaling)
pub type SscalFnPtr =
    unsafe extern "C" fn(n: *const blasint, alpha: *const f32, x: *mut f32, incx: *const blasint);

pub type SscalLp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, alpha: *const f32, x: *mut f32, incx: *const BlasInt32);

pub type SscalIlp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, alpha: *const f32, x: *mut f32, incx: *const BlasInt64);


/// Fortran dscal function pointer type (double precision vector scaling)
pub type DscalFnPtr =
    unsafe extern "C" fn(n: *const blasint, alpha: *const f64, x: *mut f64, incx: *const blasint);

pub type DscalLp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, alpha: *const f64, x: *mut f64, incx: *const BlasInt32);

pub type DscalIlp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, alpha: *const f64, x: *mut f64, incx: *const BlasInt64);


/// Fortran cscal function pointer type (single precision complex vector scaling)
pub type CscalFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const Complex32,
    x: *mut Complex32,
    incx: *const blasint,
);

pub type CscalLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    alpha: *const Complex32,
    x: *mut Complex32,
    incx: *const BlasInt32,
);

pub type CscalIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    alpha: *const Complex32,
    x: *mut Complex32,
    incx: *const BlasInt64,
);


/// Fortran zscal function pointer type (double precision complex vector scaling)
pub type ZscalFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const Complex64,
    x: *mut Complex64,
    incx: *const blasint,
);

pub type ZscalLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    alpha: *const Complex64,
    x: *mut Complex64,
    incx: *const BlasInt32,
);

pub type ZscalIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    alpha: *const Complex64,
    x: *mut Complex64,
    incx: *const BlasInt64,
);


/// Fortran csscal function pointer type (scale complex vector by real scalar)
pub type CsscalFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const f32,
    x: *mut Complex32,
    incx: *const blasint,
);

pub type CsscalLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    alpha: *const f32,
    x: *mut Complex32,
    incx: *const BlasInt32,
);

pub type CsscalIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    alpha: *const f32,
    x: *mut Complex32,
    incx: *const BlasInt64,
);


/// Fortran zdscal function pointer type (scale complex vector by real scalar)
pub type ZdscalFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const f64,
    x: *mut Complex64,
    incx: *const blasint,
);

pub type ZdscalLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    alpha: *const f64,
    x: *mut Complex64,
    incx: *const BlasInt32,
);

pub type ZdscalIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    alpha: *const f64,
    x: *mut Complex64,
    incx: *const BlasInt64,
);


/// Fortran drot function pointer type (apply Givens rotation, double precision)
pub type DrotFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut f64,
    incx: *const blasint,
    y: *mut f64,
    incy: *const blasint,
    c: *const f64,
    s: *const f64,
);

pub type DrotLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *mut f64,
    incx: *const BlasInt32,
    y: *mut f64,
    incy: *const BlasInt32,
    c: *const f64,
    s: *const f64,
);

pub type DrotIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *mut f64,
    incx: *const BlasInt64,
    y: *mut f64,
    incy: *const BlasInt64,
    c: *const f64,
    s: *const f64,
);


/// Fortran srot function pointer type (apply Givens rotation, single precision)
pub type SrotFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut f32,
    incx: *const blasint,
    y: *mut f32,
    incy: *const blasint,
    c: *const f32,
    s: *const f32,
);

pub type SrotLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *mut f32,
    incx: *const BlasInt32,
    y: *mut f32,
    incy: *const BlasInt32,
    c: *const f32,
    s: *const f32,
);

pub type SrotIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *mut f32,
    incx: *const BlasInt64,
    y: *mut f32,
    incy: *const BlasInt64,
    c: *const f32,
    s: *const f32,
);


/// Fortran drotg function pointer type (generate Givens rotation, double precision)
pub type DrotgFnPtr = unsafe extern "C" fn(a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64);

pub type DrotgLp64FnPtr = unsafe extern "C" fn(a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64);

pub type DrotgIlp64FnPtr = unsafe extern "C" fn(a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64);

/// Fortran srotg function pointer type (generate Givens rotation, single precision)
pub type SrotgFnPtr = unsafe extern "C" fn(a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32);

pub type SrotgLp64FnPtr = unsafe extern "C" fn(a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32);

pub type SrotgIlp64FnPtr = unsafe extern "C" fn(a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32);

/// Fortran drotm function pointer type (apply modified Givens rotation, double precision)
pub type DrotmFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut f64,
    incx: *const blasint,
    y: *mut f64,
    incy: *const blasint,
    p: *const f64,
);

pub type DrotmLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *mut f64,
    incx: *const BlasInt32,
    y: *mut f64,
    incy: *const BlasInt32,
    p: *const f64,
);

pub type DrotmIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *mut f64,
    incx: *const BlasInt64,
    y: *mut f64,
    incy: *const BlasInt64,
    p: *const f64,
);


/// Fortran srotm function pointer type (apply modified Givens rotation, single precision)
pub type SrotmFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut f32,
    incx: *const blasint,
    y: *mut f32,
    incy: *const blasint,
    p: *const f32,
);

pub type SrotmLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *mut f32,
    incx: *const BlasInt32,
    y: *mut f32,
    incy: *const BlasInt32,
    p: *const f32,
);

pub type SrotmIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *mut f32,
    incx: *const BlasInt64,
    y: *mut f32,
    incy: *const BlasInt64,
    p: *const f32,
);


/// Fortran drotmg function pointer type (generate modified Givens rotation, double precision)
pub type DrotmgFnPtr =
    unsafe extern "C" fn(d1: *mut f64, d2: *mut f64, b1: *mut f64, b2: *const f64, p: *mut f64);

pub type DrotmgLp64FnPtr =
    unsafe extern "C" fn(d1: *mut f64, d2: *mut f64, b1: *mut f64, b2: *const f64, p: *mut f64);

pub type DrotmgIlp64FnPtr =
    unsafe extern "C" fn(d1: *mut f64, d2: *mut f64, b1: *mut f64, b2: *const f64, p: *mut f64);


/// Fortran srotmg function pointer type (generate modified Givens rotation, single precision)
pub type SrotmgFnPtr =
    unsafe extern "C" fn(d1: *mut f32, d2: *mut f32, b1: *mut f32, b2: *const f32, p: *mut f32);

pub type SrotmgLp64FnPtr =
    unsafe extern "C" fn(d1: *mut f32, d2: *mut f32, b1: *mut f32, b2: *const f32, p: *mut f32);

pub type SrotmgIlp64FnPtr =
    unsafe extern "C" fn(d1: *mut f32, d2: *mut f32, b1: *mut f32, b2: *const f32, p: *mut f32);


/// Fortran dcabs1 function pointer type (|Re(z)| + |Im(z)|, double precision complex)
pub type Dcabs1FnPtr = unsafe extern "C" fn(z: *const Complex64) -> f64;

pub type Dcabs1Lp64FnPtr = unsafe extern "C" fn(z: *const Complex64) -> f64;

pub type Dcabs1Ilp64FnPtr = unsafe extern "C" fn(z: *const Complex64) -> f64;

/// Fortran scabs1 function pointer type (|Re(z)| + |Im(z)|, single precision complex)
pub type Scabs1FnPtr = unsafe extern "C" fn(z: *const Complex32) -> f32;

pub type Scabs1Lp64FnPtr = unsafe extern "C" fn(z: *const Complex32) -> f32;

pub type Scabs1Ilp64FnPtr = unsafe extern "C" fn(z: *const Complex32) -> f32;

/// Fortran sdot function pointer type (single precision dot product)
pub type SdotFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const f32,
    incx: *const blasint,
    y: *const f32,
    incy: *const blasint,
) -> f32;

pub type SdotLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const f32,
    incx: *const BlasInt32,
    y: *const f32,
    incy: *const BlasInt32,
) -> f32;

pub type SdotIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const f32,
    incx: *const BlasInt64,
    y: *const f32,
    incy: *const BlasInt64,
) -> f32;


/// Fortran ddot function pointer type (double precision dot product)
pub type DdotFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const f64,
    incx: *const blasint,
    y: *const f64,
    incy: *const blasint,
) -> f64;

pub type DdotLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const f64,
    incx: *const BlasInt32,
    y: *const f64,
    incy: *const BlasInt32,
) -> f64;

pub type DdotIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const f64,
    incx: *const BlasInt64,
    y: *const f64,
    incy: *const BlasInt64,
) -> f64;


/// Fortran cdotu function pointer type (complex single precision dot product, unconjugated)
/// Return value convention: complex returned via register (OpenBLAS, MKL intel, BLIS)
pub type CdotuFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    y: *const Complex32,
    incy: *const blasint,
) -> Complex32;

pub type CdotuLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const Complex32,
    incx: *const BlasInt32,
    y: *const Complex32,
    incy: *const BlasInt32,
) -> Complex32;

pub type CdotuIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const Complex32,
    incx: *const BlasInt64,
    y: *const Complex32,
    incy: *const BlasInt64,
) -> Complex32;


/// Fortran cdotu function pointer type with hidden argument convention
/// Hidden argument convention: complex written to first pointer argument (gfortran default, MKL gf)
pub type CdotuHiddenFnPtr = unsafe extern "C" fn(
    ret: *mut Complex32,
    n: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    y: *const Complex32,
    incy: *const blasint,
);

pub type CdotuHiddenLp64FnPtr = unsafe extern "C" fn(
    ret: *mut Complex32,
    n: *const BlasInt32,
    x: *const Complex32,
    incx: *const BlasInt32,
    y: *const Complex32,
    incy: *const BlasInt32,
);

pub type CdotuHiddenIlp64FnPtr = unsafe extern "C" fn(
    ret: *mut Complex32,
    n: *const BlasInt64,
    x: *const Complex32,
    incx: *const BlasInt64,
    y: *const Complex32,
    incy: *const BlasInt64,
);

/// Fortran zdotu function pointer type (complex double precision dot product, unconjugated)
/// Return value convention: complex returned via register (OpenBLAS, MKL intel, BLIS)
pub type ZdotuFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    y: *const Complex64,
    incy: *const blasint,
) -> Complex64;

pub type ZdotuLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const Complex64,
    incx: *const BlasInt32,
    y: *const Complex64,
    incy: *const BlasInt32,
) -> Complex64;

pub type ZdotuIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const Complex64,
    incx: *const BlasInt64,
    y: *const Complex64,
    incy: *const BlasInt64,
) -> Complex64;


/// Fortran zdotu function pointer type with hidden argument convention
/// Hidden argument convention: complex written to first pointer argument (gfortran default, MKL gf)
pub type ZdotuHiddenFnPtr = unsafe extern "C" fn(
    ret: *mut Complex64,
    n: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    y: *const Complex64,
    incy: *const blasint,
);

pub type ZdotuHiddenLp64FnPtr = unsafe extern "C" fn(
    ret: *mut Complex64,
    n: *const BlasInt32,
    x: *const Complex64,
    incx: *const BlasInt32,
    y: *const Complex64,
    incy: *const BlasInt32,
);

pub type ZdotuHiddenIlp64FnPtr = unsafe extern "C" fn(
    ret: *mut Complex64,
    n: *const BlasInt64,
    x: *const Complex64,
    incx: *const BlasInt64,
    y: *const Complex64,
    incy: *const BlasInt64,
);

/// Fortran cdotc function pointer type (complex single precision dot product, conjugated)
/// Return value convention: complex returned via register (OpenBLAS, MKL intel, BLIS)
pub type CdotcFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    y: *const Complex32,
    incy: *const blasint,
) -> Complex32;

pub type CdotcLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const Complex32,
    incx: *const BlasInt32,
    y: *const Complex32,
    incy: *const BlasInt32,
) -> Complex32;

pub type CdotcIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const Complex32,
    incx: *const BlasInt64,
    y: *const Complex32,
    incy: *const BlasInt64,
) -> Complex32;


/// Fortran cdotc function pointer type with hidden argument convention
/// Hidden argument convention: complex written to first pointer argument (gfortran default, MKL gf)
pub type CdotcHiddenFnPtr = unsafe extern "C" fn(
    ret: *mut Complex32,
    n: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    y: *const Complex32,
    incy: *const blasint,
);

pub type CdotcHiddenLp64FnPtr = unsafe extern "C" fn(
    ret: *mut Complex32,
    n: *const BlasInt32,
    x: *const Complex32,
    incx: *const BlasInt32,
    y: *const Complex32,
    incy: *const BlasInt32,
);

pub type CdotcHiddenIlp64FnPtr = unsafe extern "C" fn(
    ret: *mut Complex32,
    n: *const BlasInt64,
    x: *const Complex32,
    incx: *const BlasInt64,
    y: *const Complex32,
    incy: *const BlasInt64,
);

/// Fortran zdotc function pointer type (complex double precision dot product, conjugated)
/// Return value convention: complex returned via register (OpenBLAS, MKL intel, BLIS)
pub type ZdotcFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    y: *const Complex64,
    incy: *const blasint,
) -> Complex64;

pub type ZdotcLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const Complex64,
    incx: *const BlasInt32,
    y: *const Complex64,
    incy: *const BlasInt32,
) -> Complex64;

pub type ZdotcIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const Complex64,
    incx: *const BlasInt64,
    y: *const Complex64,
    incy: *const BlasInt64,
) -> Complex64;


/// Fortran zdotc function pointer type with hidden argument convention
/// Hidden argument convention: complex written to first pointer argument (gfortran default, MKL gf)
pub type ZdotcHiddenFnPtr = unsafe extern "C" fn(
    ret: *mut Complex64,
    n: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    y: *const Complex64,
    incy: *const blasint,
);

pub type ZdotcHiddenLp64FnPtr = unsafe extern "C" fn(
    ret: *mut Complex64,
    n: *const BlasInt32,
    x: *const Complex64,
    incx: *const BlasInt32,
    y: *const Complex64,
    incy: *const BlasInt32,
);

pub type ZdotcHiddenIlp64FnPtr = unsafe extern "C" fn(
    ret: *mut Complex64,
    n: *const BlasInt64,
    x: *const Complex64,
    incx: *const BlasInt64,
    y: *const Complex64,
    incy: *const BlasInt64,
);

/// Fortran sdsdot function pointer type (single precision dot product with double precision accumulation)
pub type SdsdotFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    sb: *const f32,
    x: *const f32,
    incx: *const blasint,
    y: *const f32,
    incy: *const blasint,
) -> f32;

pub type SdsdotLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    sb: *const f32,
    x: *const f32,
    incx: *const BlasInt32,
    y: *const f32,
    incy: *const BlasInt32,
) -> f32;

pub type SdsdotIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    sb: *const f32,
    x: *const f32,
    incx: *const BlasInt64,
    y: *const f32,
    incy: *const BlasInt64,
) -> f32;


/// Fortran dsdot function pointer type (double precision dot product of single precision vectors)
pub type DsdotFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const f32,
    incx: *const blasint,
    y: *const f32,
    incy: *const blasint,
) -> f64;

pub type DsdotLp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt32,
    x: *const f32,
    incx: *const BlasInt32,
    y: *const f32,
    incy: *const BlasInt32,
) -> f64;

pub type DsdotIlp64FnPtr = unsafe extern "C" fn(
    n: *const BlasInt64,
    x: *const f32,
    incx: *const BlasInt64,
    y: *const f32,
    incy: *const BlasInt64,
) -> f64;


/// Fortran snrm2 function pointer type (single precision Euclidean norm)
pub type Snrm2FnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f32, incx: *const blasint) -> f32;

pub type Snrm2Lp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const f32, incx: *const BlasInt32) -> f32;

pub type Snrm2Ilp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const f32, incx: *const BlasInt64) -> f32;


/// Fortran dnrm2 function pointer type (double precision Euclidean norm)
pub type Dnrm2FnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f64, incx: *const blasint) -> f64;

pub type Dnrm2Lp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const f64, incx: *const BlasInt32) -> f64;

pub type Dnrm2Ilp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const f64, incx: *const BlasInt64) -> f64;


/// Fortran scnrm2 function pointer type (complex single precision Euclidean norm)
pub type Scnrm2FnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex32, incx: *const blasint) -> f32;

pub type Scnrm2Lp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const Complex32, incx: *const BlasInt32) -> f32;

pub type Scnrm2Ilp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const Complex32, incx: *const BlasInt64) -> f32;


/// Fortran dznrm2 function pointer type (complex double precision Euclidean norm)
pub type Dznrm2FnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex64, incx: *const blasint) -> f64;

pub type Dznrm2Lp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const Complex64, incx: *const BlasInt32) -> f64;

pub type Dznrm2Ilp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const Complex64, incx: *const BlasInt64) -> f64;


/// Fortran sasum function pointer type (single precision sum of absolute values)
pub type SasumFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f32, incx: *const blasint) -> f32;

pub type SasumLp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const f32, incx: *const BlasInt32) -> f32;

pub type SasumIlp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const f32, incx: *const BlasInt64) -> f32;


/// Fortran dasum function pointer type (double precision sum of absolute values)
pub type DasumFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f64, incx: *const blasint) -> f64;

pub type DasumLp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const f64, incx: *const BlasInt32) -> f64;

pub type DasumIlp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const f64, incx: *const BlasInt64) -> f64;


/// Fortran scasum function pointer type (complex single precision sum of absolute values)
pub type ScasumFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex32, incx: *const blasint) -> f32;

pub type ScasumLp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const Complex32, incx: *const BlasInt32) -> f32;

pub type ScasumIlp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const Complex32, incx: *const BlasInt64) -> f32;


/// Fortran dzasum function pointer type (complex double precision sum of absolute values)
pub type DzasumFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex64, incx: *const blasint) -> f64;

pub type DzasumLp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const Complex64, incx: *const BlasInt32) -> f64;

pub type DzasumIlp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const Complex64, incx: *const BlasInt64) -> f64;


/// Fortran isamax function pointer type (index of max absolute value, single precision)
pub type IsamaxFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f32, incx: *const blasint) -> blasint;

pub type IsamaxLp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const f32, incx: *const BlasInt32) -> BlasInt32;

pub type IsamaxIlp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const f32, incx: *const BlasInt64) -> BlasInt64;


/// Fortran idamax function pointer type (index of max absolute value, double precision)
pub type IdamaxFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f64, incx: *const blasint) -> blasint;

pub type IdamaxLp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const f64, incx: *const BlasInt32) -> BlasInt32;

pub type IdamaxIlp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const f64, incx: *const BlasInt64) -> BlasInt64;


/// Fortran icamax function pointer type (index of max absolute value, complex single precision)
pub type IcamaxFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex32, incx: *const blasint) -> blasint;

pub type IcamaxLp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const Complex32, incx: *const BlasInt32) -> BlasInt32;

pub type IcamaxIlp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const Complex32, incx: *const BlasInt64) -> BlasInt64;


/// Fortran izamax function pointer type (index of max absolute value, complex double precision)
pub type IzamaxFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex64, incx: *const blasint) -> blasint;

pub type IzamaxLp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt32, x: *const Complex64, incx: *const BlasInt32) -> BlasInt32;

pub type IzamaxIlp64FnPtr =
    unsafe extern "C" fn(n: *const BlasInt64, x: *const Complex64, incx: *const BlasInt64) -> BlasInt64;


// BLAS Level 2: Matrix-Vector operations

/// Fortran sgemv function pointer type (single precision general matrix-vector multiply)
pub type SgemvFnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const blasint,
    x: *const f32,
    incx: *const blasint,
    beta: *const f32,
    y: *mut f32,
    incy: *const blasint,
);

pub type SgemvLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt32,
    x: *const f32,
    incx: *const BlasInt32,
    beta: *const f32,
    y: *mut f32,
    incy: *const BlasInt32,
);

pub type SgemvIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt64,
    x: *const f32,
    incx: *const BlasInt64,
    beta: *const f32,
    y: *mut f32,
    incy: *const BlasInt64,
);


/// Fortran dgemv function pointer type (double precision general matrix-vector multiply)
pub type DgemvFnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const blasint,
    x: *const f64,
    incx: *const blasint,
    beta: *const f64,
    y: *mut f64,
    incy: *const blasint,
);

pub type DgemvLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    x: *const f64,
    incx: *const BlasInt32,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt32,
);

pub type DgemvIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt64,
    x: *const f64,
    incx: *const BlasInt64,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt64,
);


/// Fortran cgemv function pointer type (single precision complex general matrix-vector multiply)
pub type CgemvFnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const blasint,
);

pub type CgemvLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    x: *const Complex32,
    incx: *const BlasInt32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const BlasInt32,
);

pub type CgemvIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    x: *const Complex32,
    incx: *const BlasInt64,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const BlasInt64,
);


/// Fortran zgemv function pointer type (double precision complex general matrix-vector multiply)
pub type ZgemvFnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const blasint,
);

pub type ZgemvLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    x: *const Complex64,
    incx: *const BlasInt32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const BlasInt32,
);

pub type ZgemvIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    x: *const Complex64,
    incx: *const BlasInt64,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const BlasInt64,
);


/// Fortran sgbmv function pointer type (single precision general band matrix-vector multiply)
pub type SgbmvFnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const blasint,
    n: *const blasint,
    kl: *const blasint,
    ku: *const blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const blasint,
    x: *const f32,
    incx: *const blasint,
    beta: *const f32,
    y: *mut f32,
    incy: *const blasint,
);

pub type SgbmvLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    kl: *const BlasInt32,
    ku: *const BlasInt32,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt32,
    x: *const f32,
    incx: *const BlasInt32,
    beta: *const f32,
    y: *mut f32,
    incy: *const BlasInt32,
);

pub type SgbmvIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    kl: *const BlasInt64,
    ku: *const BlasInt64,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt64,
    x: *const f32,
    incx: *const BlasInt64,
    beta: *const f32,
    y: *mut f32,
    incy: *const BlasInt64,
);


/// Fortran dgbmv function pointer type (double precision general band matrix-vector multiply)
pub type DgbmvFnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const blasint,
    n: *const blasint,
    kl: *const blasint,
    ku: *const blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const blasint,
    x: *const f64,
    incx: *const blasint,
    beta: *const f64,
    y: *mut f64,
    incy: *const blasint,
);

pub type DgbmvLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    kl: *const BlasInt32,
    ku: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    x: *const f64,
    incx: *const BlasInt32,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt32,
);

pub type DgbmvIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    kl: *const BlasInt64,
    ku: *const BlasInt64,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt64,
    x: *const f64,
    incx: *const BlasInt64,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt64,
);


/// Fortran cgbmv function pointer type (single precision complex general band matrix-vector multiply)
pub type CgbmvFnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const blasint,
    n: *const blasint,
    kl: *const blasint,
    ku: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const blasint,
);

pub type CgbmvLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    kl: *const BlasInt32,
    ku: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    x: *const Complex32,
    incx: *const BlasInt32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const BlasInt32,
);

pub type CgbmvIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    kl: *const BlasInt64,
    ku: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    x: *const Complex32,
    incx: *const BlasInt64,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const BlasInt64,
);


/// Fortran zgbmv function pointer type (double precision complex general band matrix-vector multiply)
pub type ZgbmvFnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const blasint,
    n: *const blasint,
    kl: *const blasint,
    ku: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const blasint,
);

pub type ZgbmvLp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    kl: *const BlasInt32,
    ku: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    x: *const Complex64,
    incx: *const BlasInt32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const BlasInt32,
);

pub type ZgbmvIlp64FnPtr = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    kl: *const BlasInt64,
    ku: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    x: *const Complex64,
    incx: *const BlasInt64,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const BlasInt64,
);


/// Fortran ssymv function pointer type (single precision symmetric matrix-vector multiply)
pub type SsymvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const blasint,
    x: *const f32,
    incx: *const blasint,
    beta: *const f32,
    y: *mut f32,
    incy: *const blasint,
);

pub type SsymvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt32,
    x: *const f32,
    incx: *const BlasInt32,
    beta: *const f32,
    y: *mut f32,
    incy: *const BlasInt32,
);

pub type SsymvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt64,
    x: *const f32,
    incx: *const BlasInt64,
    beta: *const f32,
    y: *mut f32,
    incy: *const BlasInt64,
);


/// Fortran dsymv function pointer type (double precision symmetric matrix-vector multiply)
pub type DsymvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const blasint,
    x: *const f64,
    incx: *const blasint,
    beta: *const f64,
    y: *mut f64,
    incy: *const blasint,
);

pub type DsymvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    x: *const f64,
    incx: *const BlasInt32,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt32,
);

pub type DsymvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt64,
    x: *const f64,
    incx: *const BlasInt64,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt64,
);


/// Fortran chemv function pointer type (single precision complex Hermitian matrix-vector multiply)
pub type ChemvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const blasint,
);

pub type ChemvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    x: *const Complex32,
    incx: *const BlasInt32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const BlasInt32,
);

pub type ChemvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    x: *const Complex32,
    incx: *const BlasInt64,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const BlasInt64,
);


/// Fortran zhemv function pointer type (double precision complex Hermitian matrix-vector multiply)
pub type ZhemvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const blasint,
);

pub type ZhemvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    x: *const Complex64,
    incx: *const BlasInt32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const BlasInt32,
);

pub type ZhemvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    x: *const Complex64,
    incx: *const BlasInt64,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const BlasInt64,
);


/// Fortran ssbmv function pointer type (single precision symmetric band matrix-vector multiply)
pub type SsbmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const blasint,
    x: *const f32,
    incx: *const blasint,
    beta: *const f32,
    y: *mut f32,
    incy: *const blasint,
);

pub type SsbmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt32,
    x: *const f32,
    incx: *const BlasInt32,
    beta: *const f32,
    y: *mut f32,
    incy: *const BlasInt32,
);

pub type SsbmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt64,
    x: *const f32,
    incx: *const BlasInt64,
    beta: *const f32,
    y: *mut f32,
    incy: *const BlasInt64,
);


/// Fortran dsbmv function pointer type (double precision symmetric band matrix-vector multiply)
pub type DsbmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const blasint,
    x: *const f64,
    incx: *const blasint,
    beta: *const f64,
    y: *mut f64,
    incy: *const blasint,
);

pub type DsbmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    x: *const f64,
    incx: *const BlasInt32,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt32,
);

pub type DsbmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt64,
    x: *const f64,
    incx: *const BlasInt64,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt64,
);


/// Fortran chbmv function pointer type (single precision complex Hermitian band matrix-vector multiply)
pub type ChbmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const blasint,
);

pub type ChbmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    x: *const Complex32,
    incx: *const BlasInt32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const BlasInt32,
);

pub type ChbmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    x: *const Complex32,
    incx: *const BlasInt64,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const BlasInt64,
);


/// Fortran zhbmv function pointer type (double precision complex Hermitian band matrix-vector multiply)
pub type ZhbmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const blasint,
);

pub type ZhbmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    x: *const Complex64,
    incx: *const BlasInt32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const BlasInt32,
);

pub type ZhbmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    x: *const Complex64,
    incx: *const BlasInt64,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const BlasInt64,
);


/// Fortran strmv function pointer type (single precision triangular matrix-vector multiply)
pub type StrmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    a: *const f32,
    lda: *const blasint,
    x: *mut f32,
    incx: *const blasint,
);

pub type StrmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    a: *const f32,
    lda: *const BlasInt32,
    x: *mut f32,
    incx: *const BlasInt32,
);

pub type StrmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    a: *const f32,
    lda: *const BlasInt64,
    x: *mut f32,
    incx: *const BlasInt64,
);


/// Fortran dtrmv function pointer type (double precision triangular matrix-vector multiply)
pub type DtrmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    a: *const f64,
    lda: *const blasint,
    x: *mut f64,
    incx: *const blasint,
);

pub type DtrmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    a: *const f64,
    lda: *const BlasInt32,
    x: *mut f64,
    incx: *const BlasInt32,
);

pub type DtrmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    a: *const f64,
    lda: *const BlasInt64,
    x: *mut f64,
    incx: *const BlasInt64,
);


/// Fortran ctrmv function pointer type (single precision complex triangular matrix-vector multiply)
pub type CtrmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    a: *const Complex32,
    lda: *const blasint,
    x: *mut Complex32,
    incx: *const blasint,
);

pub type CtrmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    a: *const Complex32,
    lda: *const BlasInt32,
    x: *mut Complex32,
    incx: *const BlasInt32,
);

pub type CtrmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    a: *const Complex32,
    lda: *const BlasInt64,
    x: *mut Complex32,
    incx: *const BlasInt64,
);


/// Fortran ztrmv function pointer type (double precision complex triangular matrix-vector multiply)
pub type ZtrmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    a: *const Complex64,
    lda: *const blasint,
    x: *mut Complex64,
    incx: *const blasint,
);

pub type ZtrmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    a: *const Complex64,
    lda: *const BlasInt32,
    x: *mut Complex64,
    incx: *const BlasInt32,
);

pub type ZtrmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    a: *const Complex64,
    lda: *const BlasInt64,
    x: *mut Complex64,
    incx: *const BlasInt64,
);


/// Fortran strsv function pointer type (single precision triangular solve)
pub type StrsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    a: *const f32,
    lda: *const blasint,
    x: *mut f32,
    incx: *const blasint,
);

pub type StrsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    a: *const f32,
    lda: *const BlasInt32,
    x: *mut f32,
    incx: *const BlasInt32,
);

pub type StrsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    a: *const f32,
    lda: *const BlasInt64,
    x: *mut f32,
    incx: *const BlasInt64,
);


/// Fortran dtrsv function pointer type (double precision triangular solve)
pub type DtrsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    a: *const f64,
    lda: *const blasint,
    x: *mut f64,
    incx: *const blasint,
);

pub type DtrsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    a: *const f64,
    lda: *const BlasInt32,
    x: *mut f64,
    incx: *const BlasInt32,
);

pub type DtrsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    a: *const f64,
    lda: *const BlasInt64,
    x: *mut f64,
    incx: *const BlasInt64,
);


/// Fortran ctrsv function pointer type (single precision complex triangular solve)
pub type CtrsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    a: *const Complex32,
    lda: *const blasint,
    x: *mut Complex32,
    incx: *const blasint,
);

pub type CtrsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    a: *const Complex32,
    lda: *const BlasInt32,
    x: *mut Complex32,
    incx: *const BlasInt32,
);

pub type CtrsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    a: *const Complex32,
    lda: *const BlasInt64,
    x: *mut Complex32,
    incx: *const BlasInt64,
);


/// Fortran ztrsv function pointer type (double precision complex triangular solve)
pub type ZtrsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    a: *const Complex64,
    lda: *const blasint,
    x: *mut Complex64,
    incx: *const blasint,
);

pub type ZtrsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    a: *const Complex64,
    lda: *const BlasInt32,
    x: *mut Complex64,
    incx: *const BlasInt32,
);

pub type ZtrsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    a: *const Complex64,
    lda: *const BlasInt64,
    x: *mut Complex64,
    incx: *const BlasInt64,
);


/// Fortran stbmv function pointer type (single precision triangular band matrix-vector multiply)
pub type StbmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    k: *const blasint,
    a: *const f32,
    lda: *const blasint,
    x: *mut f32,
    incx: *const blasint,
);

pub type StbmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    a: *const f32,
    lda: *const BlasInt32,
    x: *mut f32,
    incx: *const BlasInt32,
);

pub type StbmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    a: *const f32,
    lda: *const BlasInt64,
    x: *mut f32,
    incx: *const BlasInt64,
);


/// Fortran dtbmv function pointer type (double precision triangular band matrix-vector multiply)
pub type DtbmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    k: *const blasint,
    a: *const f64,
    lda: *const blasint,
    x: *mut f64,
    incx: *const blasint,
);

pub type DtbmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    a: *const f64,
    lda: *const BlasInt32,
    x: *mut f64,
    incx: *const BlasInt32,
);

pub type DtbmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    a: *const f64,
    lda: *const BlasInt64,
    x: *mut f64,
    incx: *const BlasInt64,
);


/// Fortran ctbmv function pointer type (single precision complex triangular band matrix-vector multiply)
pub type CtbmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    k: *const blasint,
    a: *const Complex32,
    lda: *const blasint,
    x: *mut Complex32,
    incx: *const blasint,
);

pub type CtbmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    a: *const Complex32,
    lda: *const BlasInt32,
    x: *mut Complex32,
    incx: *const BlasInt32,
);

pub type CtbmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    a: *const Complex32,
    lda: *const BlasInt64,
    x: *mut Complex32,
    incx: *const BlasInt64,
);


/// Fortran ztbmv function pointer type (double precision complex triangular band matrix-vector multiply)
pub type ZtbmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    k: *const blasint,
    a: *const Complex64,
    lda: *const blasint,
    x: *mut Complex64,
    incx: *const blasint,
);

pub type ZtbmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    a: *const Complex64,
    lda: *const BlasInt32,
    x: *mut Complex64,
    incx: *const BlasInt32,
);

pub type ZtbmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    a: *const Complex64,
    lda: *const BlasInt64,
    x: *mut Complex64,
    incx: *const BlasInt64,
);


/// Fortran stbsv function pointer type (single precision triangular band solve)
pub type StbsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    k: *const blasint,
    a: *const f32,
    lda: *const blasint,
    x: *mut f32,
    incx: *const blasint,
);

pub type StbsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    a: *const f32,
    lda: *const BlasInt32,
    x: *mut f32,
    incx: *const BlasInt32,
);

pub type StbsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    a: *const f32,
    lda: *const BlasInt64,
    x: *mut f32,
    incx: *const BlasInt64,
);


/// Fortran dtbsv function pointer type (double precision triangular band solve)
pub type DtbsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    k: *const blasint,
    a: *const f64,
    lda: *const blasint,
    x: *mut f64,
    incx: *const blasint,
);

pub type DtbsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    a: *const f64,
    lda: *const BlasInt32,
    x: *mut f64,
    incx: *const BlasInt32,
);

pub type DtbsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    a: *const f64,
    lda: *const BlasInt64,
    x: *mut f64,
    incx: *const BlasInt64,
);


/// Fortran ctbsv function pointer type (single precision complex triangular band solve)
pub type CtbsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    k: *const blasint,
    a: *const Complex32,
    lda: *const blasint,
    x: *mut Complex32,
    incx: *const blasint,
);

pub type CtbsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    a: *const Complex32,
    lda: *const BlasInt32,
    x: *mut Complex32,
    incx: *const BlasInt32,
);

pub type CtbsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    a: *const Complex32,
    lda: *const BlasInt64,
    x: *mut Complex32,
    incx: *const BlasInt64,
);


/// Fortran ztbsv function pointer type (double precision complex triangular band solve)
pub type ZtbsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    k: *const blasint,
    a: *const Complex64,
    lda: *const blasint,
    x: *mut Complex64,
    incx: *const blasint,
);

pub type ZtbsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    a: *const Complex64,
    lda: *const BlasInt32,
    x: *mut Complex64,
    incx: *const BlasInt32,
);

pub type ZtbsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    a: *const Complex64,
    lda: *const BlasInt64,
    x: *mut Complex64,
    incx: *const BlasInt64,
);


/// Fortran sger function pointer type (single precision rank-1 update)
pub type SgerFnPtr = unsafe extern "C" fn(
    m: *const blasint,
    n: *const blasint,
    alpha: *const f32,
    x: *const f32,
    incx: *const blasint,
    y: *const f32,
    incy: *const blasint,
    a: *mut f32,
    lda: *const blasint,
);

pub type SgerLp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt32,
    y: *const f32,
    incy: *const BlasInt32,
    a: *mut f32,
    lda: *const BlasInt32,
);

pub type SgerIlp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt64,
    y: *const f32,
    incy: *const BlasInt64,
    a: *mut f32,
    lda: *const BlasInt64,
);


/// Fortran dger function pointer type (double precision rank-1 update)
pub type DgerFnPtr = unsafe extern "C" fn(
    m: *const blasint,
    n: *const blasint,
    alpha: *const f64,
    x: *const f64,
    incx: *const blasint,
    y: *const f64,
    incy: *const blasint,
    a: *mut f64,
    lda: *const blasint,
);

pub type DgerLp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt32,
    y: *const f64,
    incy: *const BlasInt32,
    a: *mut f64,
    lda: *const BlasInt32,
);

pub type DgerIlp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt64,
    y: *const f64,
    incy: *const BlasInt64,
    a: *mut f64,
    lda: *const BlasInt64,
);


/// Fortran cgeru function pointer type (single complex unconjugated rank-1 update)
pub type CgeruFnPtr = unsafe extern "C" fn(
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const blasint,
    y: *const Complex32,
    incy: *const blasint,
    a: *mut Complex32,
    lda: *const blasint,
);

pub type CgeruLp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt32,
    y: *const Complex32,
    incy: *const BlasInt32,
    a: *mut Complex32,
    lda: *const BlasInt32,
);

pub type CgeruIlp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt64,
    y: *const Complex32,
    incy: *const BlasInt64,
    a: *mut Complex32,
    lda: *const BlasInt64,
);


/// Fortran cgerc function pointer type (single complex conjugated rank-1 update)
pub type CgercFnPtr = unsafe extern "C" fn(
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const blasint,
    y: *const Complex32,
    incy: *const blasint,
    a: *mut Complex32,
    lda: *const blasint,
);

pub type CgercLp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt32,
    y: *const Complex32,
    incy: *const BlasInt32,
    a: *mut Complex32,
    lda: *const BlasInt32,
);

pub type CgercIlp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt64,
    y: *const Complex32,
    incy: *const BlasInt64,
    a: *mut Complex32,
    lda: *const BlasInt64,
);


/// Fortran zgeru function pointer type (double complex unconjugated rank-1 update)
pub type ZgeruFnPtr = unsafe extern "C" fn(
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const blasint,
    y: *const Complex64,
    incy: *const blasint,
    a: *mut Complex64,
    lda: *const blasint,
);

pub type ZgeruLp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt32,
    y: *const Complex64,
    incy: *const BlasInt32,
    a: *mut Complex64,
    lda: *const BlasInt32,
);

pub type ZgeruIlp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt64,
    y: *const Complex64,
    incy: *const BlasInt64,
    a: *mut Complex64,
    lda: *const BlasInt64,
);


/// Fortran zgerc function pointer type (double complex conjugated rank-1 update)
pub type ZgercFnPtr = unsafe extern "C" fn(
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const blasint,
    y: *const Complex64,
    incy: *const blasint,
    a: *mut Complex64,
    lda: *const blasint,
);

pub type ZgercLp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt32,
    y: *const Complex64,
    incy: *const BlasInt32,
    a: *mut Complex64,
    lda: *const BlasInt32,
);

pub type ZgercIlp64FnPtr = unsafe extern "C" fn(
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt64,
    y: *const Complex64,
    incy: *const BlasInt64,
    a: *mut Complex64,
    lda: *const BlasInt64,
);


/// Fortran ssyr function pointer type (single precision symmetric rank-1 update)
pub type SsyrFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f32,
    x: *const f32,
    incx: *const blasint,
    a: *mut f32,
    lda: *const blasint,
);

pub type SsyrLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt32,
    a: *mut f32,
    lda: *const BlasInt32,
);

pub type SsyrIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt64,
    a: *mut f32,
    lda: *const BlasInt64,
);


/// Fortran dsyr function pointer type (double precision symmetric rank-1 update)
pub type DsyrFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f64,
    x: *const f64,
    incx: *const blasint,
    a: *mut f64,
    lda: *const blasint,
);

pub type DsyrLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt32,
    a: *mut f64,
    lda: *const BlasInt32,
);

pub type DsyrIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt64,
    a: *mut f64,
    lda: *const BlasInt64,
);


/// Fortran cher function pointer type (single complex hermitian rank-1 update)
pub type CherFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f32,
    x: *const Complex32,
    incx: *const blasint,
    a: *mut Complex32,
    lda: *const blasint,
);

pub type CherLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f32,
    x: *const Complex32,
    incx: *const BlasInt32,
    a: *mut Complex32,
    lda: *const BlasInt32,
);

pub type CherIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f32,
    x: *const Complex32,
    incx: *const BlasInt64,
    a: *mut Complex32,
    lda: *const BlasInt64,
);


/// Fortran zher function pointer type (double complex hermitian rank-1 update)
pub type ZherFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f64,
    x: *const Complex64,
    incx: *const blasint,
    a: *mut Complex64,
    lda: *const blasint,
);

pub type ZherLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f64,
    x: *const Complex64,
    incx: *const BlasInt32,
    a: *mut Complex64,
    lda: *const BlasInt32,
);

pub type ZherIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f64,
    x: *const Complex64,
    incx: *const BlasInt64,
    a: *mut Complex64,
    lda: *const BlasInt64,
);


/// Fortran ssyr2 function pointer type (single precision symmetric rank-2 update)
pub type Ssyr2FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f32,
    x: *const f32,
    incx: *const blasint,
    y: *const f32,
    incy: *const blasint,
    a: *mut f32,
    lda: *const blasint,
);

pub type Ssyr2Lp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt32,
    y: *const f32,
    incy: *const BlasInt32,
    a: *mut f32,
    lda: *const BlasInt32,
);

pub type Ssyr2Ilp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt64,
    y: *const f32,
    incy: *const BlasInt64,
    a: *mut f32,
    lda: *const BlasInt64,
);


/// Fortran dsyr2 function pointer type (double precision symmetric rank-2 update)
pub type Dsyr2FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f64,
    x: *const f64,
    incx: *const blasint,
    y: *const f64,
    incy: *const blasint,
    a: *mut f64,
    lda: *const blasint,
);

pub type Dsyr2Lp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt32,
    y: *const f64,
    incy: *const BlasInt32,
    a: *mut f64,
    lda: *const BlasInt32,
);

pub type Dsyr2Ilp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt64,
    y: *const f64,
    incy: *const BlasInt64,
    a: *mut f64,
    lda: *const BlasInt64,
);


/// Fortran cher2 function pointer type (single complex hermitian rank-2 update)
pub type Cher2FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const blasint,
    y: *const Complex32,
    incy: *const blasint,
    a: *mut Complex32,
    lda: *const blasint,
);

pub type Cher2Lp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt32,
    y: *const Complex32,
    incy: *const BlasInt32,
    a: *mut Complex32,
    lda: *const BlasInt32,
);

pub type Cher2Ilp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt64,
    y: *const Complex32,
    incy: *const BlasInt64,
    a: *mut Complex32,
    lda: *const BlasInt64,
);


/// Fortran zher2 function pointer type (double complex hermitian rank-2 update)
pub type Zher2FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const blasint,
    y: *const Complex64,
    incy: *const blasint,
    a: *mut Complex64,
    lda: *const blasint,
);

pub type Zher2Lp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt32,
    y: *const Complex64,
    incy: *const BlasInt32,
    a: *mut Complex64,
    lda: *const BlasInt32,
);

pub type Zher2Ilp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt64,
    y: *const Complex64,
    incy: *const BlasInt64,
    a: *mut Complex64,
    lda: *const BlasInt64,
);


// BLAS Level 2: Packed Matrix Operations

/// Fortran sspmv function pointer type (single precision symmetric packed matrix-vector multiply)
pub type SspmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f32,
    ap: *const f32,
    x: *const f32,
    incx: *const blasint,
    beta: *const f32,
    y: *mut f32,
    incy: *const blasint,
);

pub type SspmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f32,
    ap: *const f32,
    x: *const f32,
    incx: *const BlasInt32,
    beta: *const f32,
    y: *mut f32,
    incy: *const BlasInt32,
);

pub type SspmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f32,
    ap: *const f32,
    x: *const f32,
    incx: *const BlasInt64,
    beta: *const f32,
    y: *mut f32,
    incy: *const BlasInt64,
);


/// Fortran dspmv function pointer type (double precision symmetric packed matrix-vector multiply)
pub type DspmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f64,
    ap: *const f64,
    x: *const f64,
    incx: *const blasint,
    beta: *const f64,
    y: *mut f64,
    incy: *const blasint,
);

pub type DspmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f64,
    ap: *const f64,
    x: *const f64,
    incx: *const BlasInt32,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt32,
);

pub type DspmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f64,
    ap: *const f64,
    x: *const f64,
    incx: *const BlasInt64,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasInt64,
);


/// Fortran chpmv function pointer type (single precision complex Hermitian packed matrix-vector multiply)
pub type ChpmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const Complex32,
    ap: *const Complex32,
    x: *const Complex32,
    incx: *const blasint,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const blasint,
);

pub type ChpmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const Complex32,
    ap: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt32,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const BlasInt32,
);

pub type ChpmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const Complex32,
    ap: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt64,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: *const BlasInt64,
);


/// Fortran zhpmv function pointer type (double precision complex Hermitian packed matrix-vector multiply)
pub type ZhpmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const Complex64,
    ap: *const Complex64,
    x: *const Complex64,
    incx: *const blasint,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const blasint,
);

pub type ZhpmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const Complex64,
    ap: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt32,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const BlasInt32,
);

pub type ZhpmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const Complex64,
    ap: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt64,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: *const BlasInt64,
);


/// Fortran stpmv function pointer type (single precision triangular packed matrix-vector multiply)
pub type StpmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    ap: *const f32,
    x: *mut f32,
    incx: *const blasint,
);

pub type StpmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    ap: *const f32,
    x: *mut f32,
    incx: *const BlasInt32,
);

pub type StpmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    ap: *const f32,
    x: *mut f32,
    incx: *const BlasInt64,
);


/// Fortran dtpmv function pointer type (double precision triangular packed matrix-vector multiply)
pub type DtpmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    ap: *const f64,
    x: *mut f64,
    incx: *const blasint,
);

pub type DtpmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    ap: *const f64,
    x: *mut f64,
    incx: *const BlasInt32,
);

pub type DtpmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    ap: *const f64,
    x: *mut f64,
    incx: *const BlasInt64,
);


/// Fortran ctpmv function pointer type (single precision complex triangular packed matrix-vector multiply)
pub type CtpmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: *const blasint,
);

pub type CtpmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: *const BlasInt32,
);

pub type CtpmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: *const BlasInt64,
);


/// Fortran ztpmv function pointer type (double precision complex triangular packed matrix-vector multiply)
pub type ZtpmvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: *const blasint,
);

pub type ZtpmvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: *const BlasInt32,
);

pub type ZtpmvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: *const BlasInt64,
);


/// Fortran stpsv function pointer type (single precision triangular packed solve)
pub type StpsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    ap: *const f32,
    x: *mut f32,
    incx: *const blasint,
);

pub type StpsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    ap: *const f32,
    x: *mut f32,
    incx: *const BlasInt32,
);

pub type StpsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    ap: *const f32,
    x: *mut f32,
    incx: *const BlasInt64,
);


/// Fortran dtpsv function pointer type (double precision triangular packed solve)
pub type DtpsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    ap: *const f64,
    x: *mut f64,
    incx: *const blasint,
);

pub type DtpsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    ap: *const f64,
    x: *mut f64,
    incx: *const BlasInt32,
);

pub type DtpsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    ap: *const f64,
    x: *mut f64,
    incx: *const BlasInt64,
);


/// Fortran ctpsv function pointer type (single precision complex triangular packed solve)
pub type CtpsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: *const blasint,
);

pub type CtpsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: *const BlasInt32,
);

pub type CtpsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    ap: *const Complex32,
    x: *mut Complex32,
    incx: *const BlasInt64,
);


/// Fortran ztpsv function pointer type (double precision complex triangular packed solve)
pub type ZtpsvFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const blasint,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: *const blasint,
);

pub type ZtpsvLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt32,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: *const BlasInt32,
);

pub type ZtpsvIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    diag: *const c_char,
    n: *const BlasInt64,
    ap: *const Complex64,
    x: *mut Complex64,
    incx: *const BlasInt64,
);


/// Fortran sspr function pointer type (single precision symmetric packed rank-1 update)
pub type SsprFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f32,
    x: *const f32,
    incx: *const blasint,
    ap: *mut f32,
);

pub type SsprLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt32,
    ap: *mut f32,
);

pub type SsprIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt64,
    ap: *mut f32,
);


/// Fortran dspr function pointer type (double precision symmetric packed rank-1 update)
pub type DsprFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f64,
    x: *const f64,
    incx: *const blasint,
    ap: *mut f64,
);

pub type DsprLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt32,
    ap: *mut f64,
);

pub type DsprIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt64,
    ap: *mut f64,
);


/// Fortran chpr function pointer type (single precision complex Hermitian packed rank-1 update)
pub type ChprFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f32,
    x: *const Complex32,
    incx: *const blasint,
    ap: *mut Complex32,
);

pub type ChprLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f32,
    x: *const Complex32,
    incx: *const BlasInt32,
    ap: *mut Complex32,
);

pub type ChprIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f32,
    x: *const Complex32,
    incx: *const BlasInt64,
    ap: *mut Complex32,
);


/// Fortran zhpr function pointer type (double precision complex Hermitian packed rank-1 update)
pub type ZhprFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f64,
    x: *const Complex64,
    incx: *const blasint,
    ap: *mut Complex64,
);

pub type ZhprLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f64,
    x: *const Complex64,
    incx: *const BlasInt32,
    ap: *mut Complex64,
);

pub type ZhprIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f64,
    x: *const Complex64,
    incx: *const BlasInt64,
    ap: *mut Complex64,
);


/// Fortran sspr2 function pointer type (single precision symmetric packed rank-2 update)
pub type Sspr2FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f32,
    x: *const f32,
    incx: *const blasint,
    y: *const f32,
    incy: *const blasint,
    ap: *mut f32,
);

pub type Sspr2Lp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt32,
    y: *const f32,
    incy: *const BlasInt32,
    ap: *mut f32,
);

pub type Sspr2Ilp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f32,
    x: *const f32,
    incx: *const BlasInt64,
    y: *const f32,
    incy: *const BlasInt64,
    ap: *mut f32,
);


/// Fortran dspr2 function pointer type (double precision symmetric packed rank-2 update)
pub type Dspr2FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const f64,
    x: *const f64,
    incx: *const blasint,
    y: *const f64,
    incy: *const blasint,
    ap: *mut f64,
);

pub type Dspr2Lp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt32,
    y: *const f64,
    incy: *const BlasInt32,
    ap: *mut f64,
);

pub type Dspr2Ilp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const f64,
    x: *const f64,
    incx: *const BlasInt64,
    y: *const f64,
    incy: *const BlasInt64,
    ap: *mut f64,
);


/// Fortran chpr2 function pointer type (single precision complex Hermitian packed rank-2 update)
pub type Chpr2FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const blasint,
    y: *const Complex32,
    incy: *const blasint,
    ap: *mut Complex32,
);

pub type Chpr2Lp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt32,
    y: *const Complex32,
    incy: *const BlasInt32,
    ap: *mut Complex32,
);

pub type Chpr2Ilp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: *const BlasInt64,
    y: *const Complex32,
    incy: *const BlasInt64,
    ap: *mut Complex32,
);


/// Fortran zhpr2 function pointer type (double precision complex Hermitian packed rank-2 update)
pub type Zhpr2FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const blasint,
    y: *const Complex64,
    incy: *const blasint,
    ap: *mut Complex64,
);

pub type Zhpr2Lp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt32,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt32,
    y: *const Complex64,
    incy: *const BlasInt32,
    ap: *mut Complex64,
);

pub type Zhpr2Ilp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    n: *const BlasInt64,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const BlasInt64,
    y: *const Complex64,
    incy: *const BlasInt64,
    ap: *mut Complex64,
);


// BLAS Level 3: Matrix-Matrix operations

/// Fortran dgemm function pointer type (double precision general matrix multiply)
pub type DgemmFnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const blasint,
    n: *const blasint,
    k: *const blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const blasint,
    b: *const f64,
    ldb: *const blasint,
    beta: *const f64,
    c: *mut f64,
    ldc: *const blasint,
);

/// Fortran dgemm LP64 function pointer type (double precision general matrix multiply)
pub type DgemmLp64FnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    b: *const f64,
    ldb: *const BlasInt32,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt32,
);

/// Fortran dgemm ILP64 function pointer type (double precision general matrix multiply)
pub type DgemmIlp64FnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt64,
    b: *const f64,
    ldb: *const BlasInt64,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt64,
);

/// Fortran sgemm function pointer type (single precision general matrix multiply)
pub type SgemmFnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const blasint,
    n: *const blasint,
    k: *const blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const blasint,
    b: *const f32,
    ldb: *const blasint,
    beta: *const f32,
    c: *mut f32,
    ldc: *const blasint,
);

/// Fortran sgemm LP64 function pointer type (single precision general matrix multiply)
pub type SgemmLp64FnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt32,
    b: *const f32,
    ldb: *const BlasInt32,
    beta: *const f32,
    c: *mut f32,
    ldc: *const BlasInt32,
);

/// Fortran sgemm ILP64 function pointer type (single precision general matrix multiply)
pub type SgemmIlp64FnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt64,
    b: *const f32,
    ldb: *const BlasInt64,
    beta: *const f32,
    c: *mut f32,
    ldc: *const BlasInt64,
);

/// Fortran zgemm function pointer type (double precision complex general matrix multiply)
pub type ZgemmFnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const blasint,
    n: *const blasint,
    k: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    b: *const Complex64,
    ldb: *const blasint,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const blasint,
);

/// Fortran zgemm LP64 function pointer type (double precision complex general matrix multiply)
pub type ZgemmLp64FnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    b: *const Complex64,
    ldb: *const BlasInt32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const BlasInt32,
);

/// Fortran zgemm ILP64 function pointer type (double precision complex general matrix multiply)
pub type ZgemmIlp64FnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    b: *const Complex64,
    ldb: *const BlasInt64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const BlasInt64,
);

/// Fortran cgemm function pointer type (single precision complex general matrix multiply)
pub type CgemmFnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const blasint,
    n: *const blasint,
    k: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    b: *const Complex32,
    ldb: *const blasint,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const blasint,
);

/// Fortran cgemm LP64 function pointer type (single precision complex general matrix multiply)
pub type CgemmLp64FnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    b: *const Complex32,
    ldb: *const BlasInt32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const BlasInt32,
);

/// Fortran cgemm ILP64 function pointer type (single precision complex general matrix multiply)
pub type CgemmIlp64FnPtr = unsafe extern "C" fn(
    transa: *const c_char,
    transb: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    b: *const Complex32,
    ldb: *const BlasInt64,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const BlasInt64,
);

#[derive(Clone, Copy)]
pub(crate) enum DgemmProvider {
    Lp64(DgemmLp64FnPtr),
    Ilp64(DgemmIlp64FnPtr),
}

#[derive(Clone, Copy)]
pub(crate) enum SgemmProvider {
    Lp64(SgemmLp64FnPtr),
    Ilp64(SgemmIlp64FnPtr),
}

#[derive(Clone, Copy)]
pub(crate) enum ZgemmProvider {
    Lp64(ZgemmLp64FnPtr),
    Ilp64(ZgemmIlp64FnPtr),
}

#[derive(Clone, Copy)]
pub(crate) enum CgemmProvider {
    Lp64(CgemmLp64FnPtr),
    Ilp64(CgemmIlp64FnPtr),
}

/// Fortran ssymm function pointer type (single precision symmetric matrix multiply)
pub type SsymmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const blasint,
    b: *const f32,
    ldb: *const blasint,
    beta: *const f32,
    c: *mut f32,
    ldc: *const blasint,
);

pub type SsymmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt32,
    b: *const f32,
    ldb: *const BlasInt32,
    beta: *const f32,
    c: *mut f32,
    ldc: *const BlasInt32,
);

pub type SsymmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt64,
    b: *const f32,
    ldb: *const BlasInt64,
    beta: *const f32,
    c: *mut f32,
    ldc: *const BlasInt64,
);


/// Fortran dsymm function pointer type (double precision symmetric matrix multiply)
pub type DsymmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const blasint,
    b: *const f64,
    ldb: *const blasint,
    beta: *const f64,
    c: *mut f64,
    ldc: *const blasint,
);

pub type DsymmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    b: *const f64,
    ldb: *const BlasInt32,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt32,
);

pub type DsymmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt64,
    b: *const f64,
    ldb: *const BlasInt64,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt64,
);


/// Fortran csymm function pointer type (single precision complex symmetric matrix multiply)
pub type CsymmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    b: *const Complex32,
    ldb: *const blasint,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const blasint,
);

pub type CsymmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    b: *const Complex32,
    ldb: *const BlasInt32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const BlasInt32,
);

pub type CsymmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    b: *const Complex32,
    ldb: *const BlasInt64,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const BlasInt64,
);


/// Fortran zsymm function pointer type (double precision complex symmetric matrix multiply)
pub type ZsymmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    b: *const Complex64,
    ldb: *const blasint,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const blasint,
);

pub type ZsymmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    b: *const Complex64,
    ldb: *const BlasInt32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const BlasInt32,
);

pub type ZsymmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    b: *const Complex64,
    ldb: *const BlasInt64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const BlasInt64,
);


/// Fortran chemm function pointer type (single precision complex Hermitian matrix multiply)
pub type ChemmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    b: *const Complex32,
    ldb: *const blasint,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const blasint,
);

pub type ChemmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    b: *const Complex32,
    ldb: *const BlasInt32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const BlasInt32,
);

pub type ChemmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    b: *const Complex32,
    ldb: *const BlasInt64,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const BlasInt64,
);


/// Fortran zhemm function pointer type (double precision complex Hermitian matrix multiply)
pub type ZhemmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    b: *const Complex64,
    ldb: *const blasint,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const blasint,
);

pub type ZhemmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    b: *const Complex64,
    ldb: *const BlasInt32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const BlasInt32,
);

pub type ZhemmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    b: *const Complex64,
    ldb: *const BlasInt64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const BlasInt64,
);


/// Fortran dsyrk function pointer type (double precision symmetric rank-k update)
pub type DsyrkFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const blasint,
    beta: *const f64,
    c: *mut f64,
    ldc: *const blasint,
);

pub type DsyrkLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt32,
);

pub type DsyrkIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt64,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt64,
);


/// Fortran ssyrk function pointer type (single precision symmetric rank-k update)
pub type SsyrkFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const blasint,
    beta: *const f32,
    c: *mut f32,
    ldc: *const blasint,
);

pub type SsyrkLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt32,
    beta: *const f32,
    c: *mut f32,
    ldc: *const BlasInt32,
);

pub type SsyrkIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt64,
    beta: *const f32,
    c: *mut f32,
    ldc: *const BlasInt64,
);


/// Fortran csyrk function pointer type (single precision complex symmetric rank-k update)
pub type CsyrkFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const blasint,
);

pub type CsyrkLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const BlasInt32,
);

pub type CsyrkIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const BlasInt64,
);


/// Fortran zsyrk function pointer type (double precision complex symmetric rank-k update)
pub type ZsyrkFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const blasint,
);

pub type ZsyrkLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const BlasInt32,
);

pub type ZsyrkIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const BlasInt64,
);


/// Fortran cherk function pointer type (single precision complex Hermitian rank-k update)
/// Note: alpha and beta are REAL (f32), not complex
pub type CherkFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const f32,
    a: *const Complex32,
    lda: *const blasint,
    beta: *const f32,
    c: *mut Complex32,
    ldc: *const blasint,
);

pub type CherkLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f32,
    a: *const Complex32,
    lda: *const BlasInt32,
    beta: *const f32,
    c: *mut Complex32,
    ldc: *const BlasInt32,
);

pub type CherkIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const f32,
    a: *const Complex32,
    lda: *const BlasInt64,
    beta: *const f32,
    c: *mut Complex32,
    ldc: *const BlasInt64,
);


/// Fortran zherk function pointer type (double precision complex Hermitian rank-k update)
/// Note: alpha and beta are REAL (f64), not complex
pub type ZherkFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const f64,
    a: *const Complex64,
    lda: *const blasint,
    beta: *const f64,
    c: *mut Complex64,
    ldc: *const blasint,
);

pub type ZherkLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f64,
    a: *const Complex64,
    lda: *const BlasInt32,
    beta: *const f64,
    c: *mut Complex64,
    ldc: *const BlasInt32,
);

pub type ZherkIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const f64,
    a: *const Complex64,
    lda: *const BlasInt64,
    beta: *const f64,
    c: *mut Complex64,
    ldc: *const BlasInt64,
);


/// Fortran dsyr2k function pointer type (double precision symmetric rank-2k update)
pub type Dsyr2kFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const blasint,
    b: *const f64,
    ldb: *const blasint,
    beta: *const f64,
    c: *mut f64,
    ldc: *const blasint,
);

pub type Dsyr2kLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    b: *const f64,
    ldb: *const BlasInt32,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt32,
);

pub type Dsyr2kIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt64,
    b: *const f64,
    ldb: *const BlasInt64,
    beta: *const f64,
    c: *mut f64,
    ldc: *const BlasInt64,
);


/// Fortran ssyr2k function pointer type (single precision symmetric rank-2k update)
pub type Ssyr2kFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const blasint,
    b: *const f32,
    ldb: *const blasint,
    beta: *const f32,
    c: *mut f32,
    ldc: *const blasint,
);

pub type Ssyr2kLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt32,
    b: *const f32,
    ldb: *const BlasInt32,
    beta: *const f32,
    c: *mut f32,
    ldc: *const BlasInt32,
);

pub type Ssyr2kIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt64,
    b: *const f32,
    ldb: *const BlasInt64,
    beta: *const f32,
    c: *mut f32,
    ldc: *const BlasInt64,
);


/// Fortran csyr2k function pointer type (single precision complex symmetric rank-2k update)
pub type Csyr2kFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    b: *const Complex32,
    ldb: *const blasint,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const blasint,
);

pub type Csyr2kLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    b: *const Complex32,
    ldb: *const BlasInt32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const BlasInt32,
);

pub type Csyr2kIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    b: *const Complex32,
    ldb: *const BlasInt64,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const BlasInt64,
);


/// Fortran zsyr2k function pointer type (double precision complex symmetric rank-2k update)
pub type Zsyr2kFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    b: *const Complex64,
    ldb: *const blasint,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const blasint,
);

pub type Zsyr2kLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    b: *const Complex64,
    ldb: *const BlasInt32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const BlasInt32,
);

pub type Zsyr2kIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    b: *const Complex64,
    ldb: *const BlasInt64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const BlasInt64,
);


/// Fortran cher2k function pointer type (single precision complex Hermitian rank-2k update)
/// Note: alpha is complex, beta is real
pub type Cher2kFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    b: *const Complex32,
    ldb: *const blasint,
    beta: *const f32,
    c: *mut Complex32,
    ldc: *const blasint,
);

pub type Cher2kLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    b: *const Complex32,
    ldb: *const BlasInt32,
    beta: *const f32,
    c: *mut Complex32,
    ldc: *const BlasInt32,
);

pub type Cher2kIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    b: *const Complex32,
    ldb: *const BlasInt64,
    beta: *const f32,
    c: *mut Complex32,
    ldc: *const BlasInt64,
);


/// Fortran zher2k function pointer type (double precision complex Hermitian rank-2k update)
/// Note: alpha is complex, beta is real
pub type Zher2kFnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const blasint,
    k: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    b: *const Complex64,
    ldb: *const blasint,
    beta: *const f64,
    c: *mut Complex64,
    ldc: *const blasint,
);

pub type Zher2kLp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    b: *const Complex64,
    ldb: *const BlasInt32,
    beta: *const f64,
    c: *mut Complex64,
    ldc: *const BlasInt32,
);

pub type Zher2kIlp64FnPtr = unsafe extern "C" fn(
    uplo: *const c_char,
    trans: *const c_char,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    b: *const Complex64,
    ldb: *const BlasInt64,
    beta: *const f64,
    c: *mut Complex64,
    ldc: *const BlasInt64,
);


/// Fortran dtrmm function pointer type (double precision triangular matrix multiply)
pub type DtrmmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const blasint,
    b: *mut f64,
    ldb: *const blasint,
);

pub type DtrmmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    b: *mut f64,
    ldb: *const BlasInt32,
);

pub type DtrmmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt64,
    b: *mut f64,
    ldb: *const BlasInt64,
);


/// Fortran dtrsm function pointer type (double precision triangular solve)
pub type DtrsmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const f64,
    a: *const f64,
    lda: *const blasint,
    b: *mut f64,
    ldb: *const blasint,
);

pub type DtrsmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt32,
    b: *mut f64,
    ldb: *const BlasInt32,
);

pub type DtrsmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasInt64,
    b: *mut f64,
    ldb: *const BlasInt64,
);


/// Fortran strmm function pointer type (single precision triangular matrix multiply)
pub type StrmmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const blasint,
    b: *mut f32,
    ldb: *const blasint,
);

pub type StrmmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt32,
    b: *mut f32,
    ldb: *const BlasInt32,
);

pub type StrmmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt64,
    b: *mut f32,
    ldb: *const BlasInt64,
);


/// Fortran ctrmm function pointer type (single precision complex triangular matrix multiply)
pub type CtrmmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    b: *mut Complex32,
    ldb: *const blasint,
);

pub type CtrmmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    b: *mut Complex32,
    ldb: *const BlasInt32,
);

pub type CtrmmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    b: *mut Complex32,
    ldb: *const BlasInt64,
);


/// Fortran ztrmm function pointer type (double precision complex triangular matrix multiply)
pub type ZtrmmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    b: *mut Complex64,
    ldb: *const blasint,
);

pub type ZtrmmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    b: *mut Complex64,
    ldb: *const BlasInt32,
);

pub type ZtrmmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    b: *mut Complex64,
    ldb: *const BlasInt64,
);


/// Fortran strsm function pointer type (single precision triangular solve)
pub type StrsmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const f32,
    a: *const f32,
    lda: *const blasint,
    b: *mut f32,
    ldb: *const blasint,
);

pub type StrsmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt32,
    b: *mut f32,
    ldb: *const BlasInt32,
);

pub type StrsmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const f32,
    a: *const f32,
    lda: *const BlasInt64,
    b: *mut f32,
    ldb: *const BlasInt64,
);


/// Fortran ctrsm function pointer type (single precision complex triangular solve)
pub type CtrsmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const blasint,
    b: *mut Complex32,
    ldb: *const blasint,
);

pub type CtrsmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt32,
    b: *mut Complex32,
    ldb: *const BlasInt32,
);

pub type CtrsmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const BlasInt64,
    b: *mut Complex32,
    ldb: *const BlasInt64,
);


/// Fortran ztrsm function pointer type (double precision complex triangular solve)
pub type ZtrsmFnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const blasint,
    n: *const blasint,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const blasint,
    b: *mut Complex64,
    ldb: *const blasint,
);

pub type ZtrsmLp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt32,
    b: *mut Complex64,
    ldb: *const BlasInt32,
);

pub type ZtrsmIlp64FnPtr = unsafe extern "C" fn(
    side: *const c_char,
    uplo: *const c_char,
    transa: *const c_char,
    diag: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const BlasInt64,
    b: *mut Complex64,
    ldb: *const BlasInt64,
);


// =============================================================================
// Complex return style configuration
// =============================================================================

use crate::types::ComplexReturnStyle;

/// Global complex return style setting.
/// Must be set before registering cdotu, zdotu, cdotc, zdotc.
static COMPLEX_RETURN_STYLE: OnceLock<ComplexReturnStyle> = OnceLock::new();

/// Set the complex return style for Fortran BLAS functions.
///
/// This must be called before registering cdotu, zdotu, cdotc, zdotc.
///
/// # Safety
///
/// Must be called before any complex dot product functions are registered.
///
/// # Panics
///
/// Panics if the style has already been set.
#[no_mangle]
pub unsafe extern "C" fn set_complex_return_style(style: ComplexReturnStyle) {
    COMPLEX_RETURN_STYLE
        .set(style)
        .expect("complex return style already set (can only be set once)");
}

/// Get the current complex return style.
///
/// Returns `ReturnValue` as the default if not explicitly set.
#[no_mangle]
pub extern "C" fn get_complex_return_style() -> ComplexReturnStyle {
    COMPLEX_RETURN_STYLE
        .get()
        .copied()
        .unwrap_or(ComplexReturnStyle::ReturnValue)
}

// =============================================================================
// Function pointer storage (OnceLock per function)
// =============================================================================

// BLAS Level 1
static SSWAP: OnceLock<SswapFnPtr> = OnceLock::new();
static DSWAP: OnceLock<DswapFnPtr> = OnceLock::new();
static CSWAP: OnceLock<CswapFnPtr> = OnceLock::new();
static ZSWAP: OnceLock<ZswapFnPtr> = OnceLock::new();
static SCOPY: OnceLock<ScopyFnPtr> = OnceLock::new();
static DCOPY: OnceLock<DcopyFnPtr> = OnceLock::new();
static CCOPY: OnceLock<CcopyFnPtr> = OnceLock::new();
static ZCOPY: OnceLock<ZcopyFnPtr> = OnceLock::new();
static SAXPY: OnceLock<SaxpyFnPtr> = OnceLock::new();
static DAXPY: OnceLock<DaxpyFnPtr> = OnceLock::new();
static CAXPY: OnceLock<CaxpyFnPtr> = OnceLock::new();
static ZAXPY: OnceLock<ZaxpyFnPtr> = OnceLock::new();
static SSCAL: OnceLock<SscalFnPtr> = OnceLock::new();
static DSCAL: OnceLock<DscalFnPtr> = OnceLock::new();
static CSCAL: OnceLock<CscalFnPtr> = OnceLock::new();
static ZSCAL: OnceLock<ZscalFnPtr> = OnceLock::new();
static CSSCAL: OnceLock<CsscalFnPtr> = OnceLock::new();
static ZDSCAL: OnceLock<ZdscalFnPtr> = OnceLock::new();
static SROT: OnceLock<SrotFnPtr> = OnceLock::new();
static DROT: OnceLock<DrotFnPtr> = OnceLock::new();
static SROTG: OnceLock<SrotgFnPtr> = OnceLock::new();
static DROTG: OnceLock<DrotgFnPtr> = OnceLock::new();
static SROTM: OnceLock<SrotmFnPtr> = OnceLock::new();
static DROTM: OnceLock<DrotmFnPtr> = OnceLock::new();
static SROTMG: OnceLock<SrotmgFnPtr> = OnceLock::new();
static DROTMG: OnceLock<DrotmgFnPtr> = OnceLock::new();
static SCABS1: OnceLock<Scabs1FnPtr> = OnceLock::new();
static DCABS1: OnceLock<Dcabs1FnPtr> = OnceLock::new();
static SDOT: OnceLock<SdotFnPtr> = OnceLock::new();
static DDOT: OnceLock<DdotFnPtr> = OnceLock::new();
// Complex dot products use raw pointers to support dual ABI conventions
// We use a wrapper type to make the pointers Sync+Send (they are function pointers, which are safe to share)
#[derive(Clone, Copy, Debug)]
struct FnPtrWrapper(*const ());

// Safety: Function pointers are safe to share between threads
unsafe impl Sync for FnPtrWrapper {}
unsafe impl Send for FnPtrWrapper {}

static CDOTU_LP64_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static CDOTU_ILP64_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static ZDOTU_LP64_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static ZDOTU_ILP64_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static CDOTC_LP64_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static CDOTC_ILP64_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static ZDOTC_LP64_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static ZDOTC_ILP64_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static SDSDOT: OnceLock<SdsdotFnPtr> = OnceLock::new();
static DSDOT: OnceLock<DsdotFnPtr> = OnceLock::new();
static SNRM2: OnceLock<Snrm2FnPtr> = OnceLock::new();
static DNRM2: OnceLock<Dnrm2FnPtr> = OnceLock::new();
static SCNRM2: OnceLock<Scnrm2FnPtr> = OnceLock::new();
static DZNRM2: OnceLock<Dznrm2FnPtr> = OnceLock::new();
static SASUM: OnceLock<SasumFnPtr> = OnceLock::new();
static DASUM: OnceLock<DasumFnPtr> = OnceLock::new();
static SCASUM: OnceLock<ScasumFnPtr> = OnceLock::new();
static DZASUM: OnceLock<DzasumFnPtr> = OnceLock::new();
static ISAMAX: OnceLock<IsamaxFnPtr> = OnceLock::new();
static IDAMAX: OnceLock<IdamaxFnPtr> = OnceLock::new();
static ICAMAX: OnceLock<IcamaxFnPtr> = OnceLock::new();
static IZAMAX: OnceLock<IzamaxFnPtr> = OnceLock::new();

// BLAS Level 2
static SGEMV: OnceLock<SgemvFnPtr> = OnceLock::new();
static DGEMV: OnceLock<DgemvFnPtr> = OnceLock::new();
static CGEMV: OnceLock<CgemvFnPtr> = OnceLock::new();
static ZGEMV: OnceLock<ZgemvFnPtr> = OnceLock::new();
static SGBMV: OnceLock<SgbmvFnPtr> = OnceLock::new();
static DGBMV: OnceLock<DgbmvFnPtr> = OnceLock::new();
static CGBMV: OnceLock<CgbmvFnPtr> = OnceLock::new();
static ZGBMV: OnceLock<ZgbmvFnPtr> = OnceLock::new();
static SSYMV: OnceLock<SsymvFnPtr> = OnceLock::new();
static DSYMV: OnceLock<DsymvFnPtr> = OnceLock::new();
static CHEMV: OnceLock<ChemvFnPtr> = OnceLock::new();
static ZHEMV: OnceLock<ZhemvFnPtr> = OnceLock::new();
static SSBMV: OnceLock<SsbmvFnPtr> = OnceLock::new();
static DSBMV: OnceLock<DsbmvFnPtr> = OnceLock::new();
static CHBMV: OnceLock<ChbmvFnPtr> = OnceLock::new();
static ZHBMV: OnceLock<ZhbmvFnPtr> = OnceLock::new();
static STRMV: OnceLock<StrmvFnPtr> = OnceLock::new();
static DTRMV: OnceLock<DtrmvFnPtr> = OnceLock::new();
static CTRMV: OnceLock<CtrmvFnPtr> = OnceLock::new();
static ZTRMV: OnceLock<ZtrmvFnPtr> = OnceLock::new();
static STRSV: OnceLock<StrsvFnPtr> = OnceLock::new();
static DTRSV: OnceLock<DtrsvFnPtr> = OnceLock::new();
static CTRSV: OnceLock<CtrsvFnPtr> = OnceLock::new();
static ZTRSV: OnceLock<ZtrsvFnPtr> = OnceLock::new();
static STBMV: OnceLock<StbmvFnPtr> = OnceLock::new();
static DTBMV: OnceLock<DtbmvFnPtr> = OnceLock::new();
static CTBMV: OnceLock<CtbmvFnPtr> = OnceLock::new();
static ZTBMV: OnceLock<ZtbmvFnPtr> = OnceLock::new();
static STBSV: OnceLock<StbsvFnPtr> = OnceLock::new();
static DTBSV: OnceLock<DtbsvFnPtr> = OnceLock::new();
static CTBSV: OnceLock<CtbsvFnPtr> = OnceLock::new();
static ZTBSV: OnceLock<ZtbsvFnPtr> = OnceLock::new();
static SGER: OnceLock<SgerFnPtr> = OnceLock::new();
static DGER: OnceLock<DgerFnPtr> = OnceLock::new();
static CGERU: OnceLock<CgeruFnPtr> = OnceLock::new();
static CGERC: OnceLock<CgercFnPtr> = OnceLock::new();
static ZGERU: OnceLock<ZgeruFnPtr> = OnceLock::new();
static ZGERC: OnceLock<ZgercFnPtr> = OnceLock::new();
static SSYR: OnceLock<SsyrFnPtr> = OnceLock::new();
static DSYR: OnceLock<DsyrFnPtr> = OnceLock::new();
static CHER: OnceLock<CherFnPtr> = OnceLock::new();
static ZHER: OnceLock<ZherFnPtr> = OnceLock::new();
static SSYR2: OnceLock<Ssyr2FnPtr> = OnceLock::new();
static DSYR2: OnceLock<Dsyr2FnPtr> = OnceLock::new();
static CHER2: OnceLock<Cher2FnPtr> = OnceLock::new();
static ZHER2: OnceLock<Zher2FnPtr> = OnceLock::new();

// BLAS Level 2 - Packed matrix operations
static SSPMV: OnceLock<SspmvFnPtr> = OnceLock::new();
static DSPMV: OnceLock<DspmvFnPtr> = OnceLock::new();
static CHPMV: OnceLock<ChpmvFnPtr> = OnceLock::new();
static ZHPMV: OnceLock<ZhpmvFnPtr> = OnceLock::new();
static STPMV: OnceLock<StpmvFnPtr> = OnceLock::new();
static DTPMV: OnceLock<DtpmvFnPtr> = OnceLock::new();
static CTPMV: OnceLock<CtpmvFnPtr> = OnceLock::new();
static ZTPMV: OnceLock<ZtpmvFnPtr> = OnceLock::new();
static STPSV: OnceLock<StpsvFnPtr> = OnceLock::new();
static DTPSV: OnceLock<DtpsvFnPtr> = OnceLock::new();
static CTPSV: OnceLock<CtpsvFnPtr> = OnceLock::new();
static ZTPSV: OnceLock<ZtpsvFnPtr> = OnceLock::new();
static SSPR: OnceLock<SsprFnPtr> = OnceLock::new();
static DSPR: OnceLock<DsprFnPtr> = OnceLock::new();
static CHPR: OnceLock<ChprFnPtr> = OnceLock::new();
static ZHPR: OnceLock<ZhprFnPtr> = OnceLock::new();
static SSPR2: OnceLock<Sspr2FnPtr> = OnceLock::new();
static DSPR2: OnceLock<Dspr2FnPtr> = OnceLock::new();
static CHPR2: OnceLock<Chpr2FnPtr> = OnceLock::new();
static ZHPR2: OnceLock<Zhpr2FnPtr> = OnceLock::new();

// BLAS Level 3
static DGEMM: OnceLock<DgemmFnPtr> = OnceLock::new();
static DGEMM_LP64: OnceLock<DgemmLp64FnPtr> = OnceLock::new();
static DGEMM_ILP64: OnceLock<DgemmIlp64FnPtr> = OnceLock::new();
static SGEMM: OnceLock<SgemmFnPtr> = OnceLock::new();
static SGEMM_LP64: OnceLock<SgemmLp64FnPtr> = OnceLock::new();
static SGEMM_ILP64: OnceLock<SgemmIlp64FnPtr> = OnceLock::new();
static ZGEMM: OnceLock<ZgemmFnPtr> = OnceLock::new();
static ZGEMM_LP64: OnceLock<ZgemmLp64FnPtr> = OnceLock::new();
static ZGEMM_ILP64: OnceLock<ZgemmIlp64FnPtr> = OnceLock::new();
static CGEMM: OnceLock<CgemmFnPtr> = OnceLock::new();
static CGEMM_LP64: OnceLock<CgemmLp64FnPtr> = OnceLock::new();
static CGEMM_ILP64: OnceLock<CgemmIlp64FnPtr> = OnceLock::new();
static SSYMM: OnceLock<SsymmFnPtr> = OnceLock::new();
static DSYMM: OnceLock<DsymmFnPtr> = OnceLock::new();
static CSYMM: OnceLock<CsymmFnPtr> = OnceLock::new();
static ZSYMM: OnceLock<ZsymmFnPtr> = OnceLock::new();
static CHEMM: OnceLock<ChemmFnPtr> = OnceLock::new();
static ZHEMM: OnceLock<ZhemmFnPtr> = OnceLock::new();
static DSYRK: OnceLock<DsyrkFnPtr> = OnceLock::new();
static SSYRK: OnceLock<SsyrkFnPtr> = OnceLock::new();
static CSYRK: OnceLock<CsyrkFnPtr> = OnceLock::new();
static ZSYRK: OnceLock<ZsyrkFnPtr> = OnceLock::new();
static CHERK: OnceLock<CherkFnPtr> = OnceLock::new();
static ZHERK: OnceLock<ZherkFnPtr> = OnceLock::new();
static DSYR2K: OnceLock<Dsyr2kFnPtr> = OnceLock::new();
static SSYR2K: OnceLock<Ssyr2kFnPtr> = OnceLock::new();
static CSYR2K: OnceLock<Csyr2kFnPtr> = OnceLock::new();
static ZSYR2K: OnceLock<Zsyr2kFnPtr> = OnceLock::new();
static CHER2K: OnceLock<Cher2kFnPtr> = OnceLock::new();
static ZHER2K: OnceLock<Zher2kFnPtr> = OnceLock::new();
static DTRMM: OnceLock<DtrmmFnPtr> = OnceLock::new();
static DTRSM: OnceLock<DtrsmFnPtr> = OnceLock::new();
static STRMM: OnceLock<StrmmFnPtr> = OnceLock::new();
static CTRMM: OnceLock<CtrmmFnPtr> = OnceLock::new();
static ZTRMM: OnceLock<ZtrmmFnPtr> = OnceLock::new();
static STRSM: OnceLock<StrsmFnPtr> = OnceLock::new();
static CTRSM: OnceLock<CtrsmFnPtr> = OnceLock::new();
static ZTRSM: OnceLock<ZtrsmFnPtr> = OnceLock::new();

static REGISTRATION_LOCK: Mutex<()> = Mutex::new(());

// Dual LP64/ILP64 backend definitions
define_dual_backend!(Sswap, "sswap", SswapLp64FnPtr, SswapIlp64FnPtr, SswapProvider);
define_dual_backend!(Dswap, "dswap", DswapLp64FnPtr, DswapIlp64FnPtr, DswapProvider);
define_dual_backend!(Cswap, "cswap", CswapLp64FnPtr, CswapIlp64FnPtr, CswapProvider);
define_dual_backend!(Zswap, "zswap", ZswapLp64FnPtr, ZswapIlp64FnPtr, ZswapProvider);
define_dual_backend!(Scopy, "scopy", ScopyLp64FnPtr, ScopyIlp64FnPtr, ScopyProvider);
define_dual_backend!(Dcopy, "dcopy", DcopyLp64FnPtr, DcopyIlp64FnPtr, DcopyProvider);
define_dual_backend!(Ccopy, "ccopy", CcopyLp64FnPtr, CcopyIlp64FnPtr, CcopyProvider);
define_dual_backend!(Zcopy, "zcopy", ZcopyLp64FnPtr, ZcopyIlp64FnPtr, ZcopyProvider);
define_dual_backend!(Saxpy, "saxpy", SaxpyLp64FnPtr, SaxpyIlp64FnPtr, SaxpyProvider);
define_dual_backend!(Daxpy, "daxpy", DaxpyLp64FnPtr, DaxpyIlp64FnPtr, DaxpyProvider);
define_dual_backend!(Caxpy, "caxpy", CaxpyLp64FnPtr, CaxpyIlp64FnPtr, CaxpyProvider);
define_dual_backend!(Zaxpy, "zaxpy", ZaxpyLp64FnPtr, ZaxpyIlp64FnPtr, ZaxpyProvider);
define_dual_backend!(Sscal, "sscal", SscalLp64FnPtr, SscalIlp64FnPtr, SscalProvider);
define_dual_backend!(Dscal, "dscal", DscalLp64FnPtr, DscalIlp64FnPtr, DscalProvider);
define_dual_backend!(Cscal, "cscal", CscalLp64FnPtr, CscalIlp64FnPtr, CscalProvider);
define_dual_backend!(Zscal, "zscal", ZscalLp64FnPtr, ZscalIlp64FnPtr, ZscalProvider);
define_dual_backend!(Csscal, "csscal", CsscalLp64FnPtr, CsscalIlp64FnPtr, CsscalProvider);
define_dual_backend!(Zdscal, "zdscal", ZdscalLp64FnPtr, ZdscalIlp64FnPtr, ZdscalProvider);
define_dual_backend!(Srot, "srot", SrotLp64FnPtr, SrotIlp64FnPtr, SrotProvider);
define_dual_backend!(Drot, "drot", DrotLp64FnPtr, DrotIlp64FnPtr, DrotProvider);
define_dual_backend!(Srotg, "srotg", SrotgLp64FnPtr, SrotgIlp64FnPtr, SrotgProvider);
define_dual_backend!(Drotg, "drotg", DrotgLp64FnPtr, DrotgIlp64FnPtr, DrotgProvider);
define_dual_backend!(Srotm, "srotm", SrotmLp64FnPtr, SrotmIlp64FnPtr, SrotmProvider);
define_dual_backend!(Drotm, "drotm", DrotmLp64FnPtr, DrotmIlp64FnPtr, DrotmProvider);
define_dual_backend!(Srotmg, "srotmg", SrotmgLp64FnPtr, SrotmgIlp64FnPtr, SrotmgProvider);
define_dual_backend!(Drotmg, "drotmg", DrotmgLp64FnPtr, DrotmgIlp64FnPtr, DrotmgProvider);
define_dual_backend!(Scabs1, "scabs1", Scabs1Lp64FnPtr, Scabs1Ilp64FnPtr, Scabs1Provider);
define_dual_backend!(Dcabs1, "dcabs1", Dcabs1Lp64FnPtr, Dcabs1Ilp64FnPtr, Dcabs1Provider);
define_dual_backend!(Sdot, "sdot", SdotLp64FnPtr, SdotIlp64FnPtr, SdotProvider);
define_dual_backend!(Ddot, "ddot", DdotLp64FnPtr, DdotIlp64FnPtr, DdotProvider);
define_dual_backend!(Sdsdot, "sdsdot", SdsdotLp64FnPtr, SdsdotIlp64FnPtr, SdsdotProvider);
define_dual_backend!(Dsdot, "dsdot", DsdotLp64FnPtr, DsdotIlp64FnPtr, DsdotProvider);
define_dual_backend!(Snrm2, "snrm2", Snrm2Lp64FnPtr, Snrm2Ilp64FnPtr, Snrm2Provider);
define_dual_backend!(Dnrm2, "dnrm2", Dnrm2Lp64FnPtr, Dnrm2Ilp64FnPtr, Dnrm2Provider);
define_dual_backend!(Scnrm2, "scnrm2", Scnrm2Lp64FnPtr, Scnrm2Ilp64FnPtr, Scnrm2Provider);
define_dual_backend!(Dznrm2, "dznrm2", Dznrm2Lp64FnPtr, Dznrm2Ilp64FnPtr, Dznrm2Provider);
define_dual_backend!(Sasum, "sasum", SasumLp64FnPtr, SasumIlp64FnPtr, SasumProvider);
define_dual_backend!(Dasum, "dasum", DasumLp64FnPtr, DasumIlp64FnPtr, DasumProvider);
define_dual_backend!(Scasum, "scasum", ScasumLp64FnPtr, ScasumIlp64FnPtr, ScasumProvider);
define_dual_backend!(Dzasum, "dzasum", DzasumLp64FnPtr, DzasumIlp64FnPtr, DzasumProvider);
define_dual_backend!(Isamax, "isamax", IsamaxLp64FnPtr, IsamaxIlp64FnPtr, IsamaxProvider);
define_dual_backend!(Idamax, "idamax", IdamaxLp64FnPtr, IdamaxIlp64FnPtr, IdamaxProvider);
define_dual_backend!(Icamax, "icamax", IcamaxLp64FnPtr, IcamaxIlp64FnPtr, IcamaxProvider);
define_dual_backend!(Izamax, "izamax", IzamaxLp64FnPtr, IzamaxIlp64FnPtr, IzamaxProvider);
define_dual_backend!(Sgemv, "sgemv", SgemvLp64FnPtr, SgemvIlp64FnPtr, SgemvProvider);
define_dual_backend!(Dgemv, "dgemv", DgemvLp64FnPtr, DgemvIlp64FnPtr, DgemvProvider);
define_dual_backend!(Cgemv, "cgemv", CgemvLp64FnPtr, CgemvIlp64FnPtr, CgemvProvider);
define_dual_backend!(Zgemv, "zgemv", ZgemvLp64FnPtr, ZgemvIlp64FnPtr, ZgemvProvider);
define_dual_backend!(Sgbmv, "sgbmv", SgbmvLp64FnPtr, SgbmvIlp64FnPtr, SgbmvProvider);
define_dual_backend!(Dgbmv, "dgbmv", DgbmvLp64FnPtr, DgbmvIlp64FnPtr, DgbmvProvider);
define_dual_backend!(Cgbmv, "cgbmv", CgbmvLp64FnPtr, CgbmvIlp64FnPtr, CgbmvProvider);
define_dual_backend!(Zgbmv, "zgbmv", ZgbmvLp64FnPtr, ZgbmvIlp64FnPtr, ZgbmvProvider);
define_dual_backend!(Ssymv, "ssymv", SsymvLp64FnPtr, SsymvIlp64FnPtr, SsymvProvider);
define_dual_backend!(Dsymv, "dsymv", DsymvLp64FnPtr, DsymvIlp64FnPtr, DsymvProvider);
define_dual_backend!(Chemv, "chemv", ChemvLp64FnPtr, ChemvIlp64FnPtr, ChemvProvider);
define_dual_backend!(Zhemv, "zhemv", ZhemvLp64FnPtr, ZhemvIlp64FnPtr, ZhemvProvider);
define_dual_backend!(Ssbmv, "ssbmv", SsbmvLp64FnPtr, SsbmvIlp64FnPtr, SsbmvProvider);
define_dual_backend!(Dsbmv, "dsbmv", DsbmvLp64FnPtr, DsbmvIlp64FnPtr, DsbmvProvider);
define_dual_backend!(Chbmv, "chbmv", ChbmvLp64FnPtr, ChbmvIlp64FnPtr, ChbmvProvider);
define_dual_backend!(Zhbmv, "zhbmv", ZhbmvLp64FnPtr, ZhbmvIlp64FnPtr, ZhbmvProvider);
define_dual_backend!(Strmv, "strmv", StrmvLp64FnPtr, StrmvIlp64FnPtr, StrmvProvider);
define_dual_backend!(Dtrmv, "dtrmv", DtrmvLp64FnPtr, DtrmvIlp64FnPtr, DtrmvProvider);
define_dual_backend!(Ctrmv, "ctrmv", CtrmvLp64FnPtr, CtrmvIlp64FnPtr, CtrmvProvider);
define_dual_backend!(Ztrmv, "ztrmv", ZtrmvLp64FnPtr, ZtrmvIlp64FnPtr, ZtrmvProvider);
define_dual_backend!(Strsv, "strsv", StrsvLp64FnPtr, StrsvIlp64FnPtr, StrsvProvider);
define_dual_backend!(Dtrsv, "dtrsv", DtrsvLp64FnPtr, DtrsvIlp64FnPtr, DtrsvProvider);
define_dual_backend!(Ctrsv, "ctrsv", CtrsvLp64FnPtr, CtrsvIlp64FnPtr, CtrsvProvider);
define_dual_backend!(Ztrsv, "ztrsv", ZtrsvLp64FnPtr, ZtrsvIlp64FnPtr, ZtrsvProvider);
define_dual_backend!(Stbmv, "stbmv", StbmvLp64FnPtr, StbmvIlp64FnPtr, StbmvProvider);
define_dual_backend!(Dtbmv, "dtbmv", DtbmvLp64FnPtr, DtbmvIlp64FnPtr, DtbmvProvider);
define_dual_backend!(Ctbmv, "ctbmv", CtbmvLp64FnPtr, CtbmvIlp64FnPtr, CtbmvProvider);
define_dual_backend!(Ztbmv, "ztbmv", ZtbmvLp64FnPtr, ZtbmvIlp64FnPtr, ZtbmvProvider);
define_dual_backend!(Stbsv, "stbsv", StbsvLp64FnPtr, StbsvIlp64FnPtr, StbsvProvider);
define_dual_backend!(Dtbsv, "dtbsv", DtbsvLp64FnPtr, DtbsvIlp64FnPtr, DtbsvProvider);
define_dual_backend!(Ctbsv, "ctbsv", CtbsvLp64FnPtr, CtbsvIlp64FnPtr, CtbsvProvider);
define_dual_backend!(Ztbsv, "ztbsv", ZtbsvLp64FnPtr, ZtbsvIlp64FnPtr, ZtbsvProvider);
define_dual_backend!(Sger, "sger", SgerLp64FnPtr, SgerIlp64FnPtr, SgerProvider);
define_dual_backend!(Dger, "dger", DgerLp64FnPtr, DgerIlp64FnPtr, DgerProvider);
define_dual_backend!(Cgeru, "cgeru", CgeruLp64FnPtr, CgeruIlp64FnPtr, CgeruProvider);
define_dual_backend!(Cgerc, "cgerc", CgercLp64FnPtr, CgercIlp64FnPtr, CgercProvider);
define_dual_backend!(Zgeru, "zgeru", ZgeruLp64FnPtr, ZgeruIlp64FnPtr, ZgeruProvider);
define_dual_backend!(Zgerc, "zgerc", ZgercLp64FnPtr, ZgercIlp64FnPtr, ZgercProvider);
define_dual_backend!(Ssyr, "ssyr", SsyrLp64FnPtr, SsyrIlp64FnPtr, SsyrProvider);
define_dual_backend!(Dsyr, "dsyr", DsyrLp64FnPtr, DsyrIlp64FnPtr, DsyrProvider);
define_dual_backend!(Cher, "cher", CherLp64FnPtr, CherIlp64FnPtr, CherProvider);
define_dual_backend!(Zher, "zher", ZherLp64FnPtr, ZherIlp64FnPtr, ZherProvider);
define_dual_backend!(Ssyr2, "ssyr2", Ssyr2Lp64FnPtr, Ssyr2Ilp64FnPtr, Ssyr2Provider);
define_dual_backend!(Dsyr2, "dsyr2", Dsyr2Lp64FnPtr, Dsyr2Ilp64FnPtr, Dsyr2Provider);
define_dual_backend!(Cher2, "cher2", Cher2Lp64FnPtr, Cher2Ilp64FnPtr, Cher2Provider);
define_dual_backend!(Zher2, "zher2", Zher2Lp64FnPtr, Zher2Ilp64FnPtr, Zher2Provider);
define_dual_backend!(Sspmv, "sspmv", SspmvLp64FnPtr, SspmvIlp64FnPtr, SspmvProvider);
define_dual_backend!(Dspmv, "dspmv", DspmvLp64FnPtr, DspmvIlp64FnPtr, DspmvProvider);
define_dual_backend!(Chpmv, "chpmv", ChpmvLp64FnPtr, ChpmvIlp64FnPtr, ChpmvProvider);
define_dual_backend!(Zhpmv, "zhpmv", ZhpmvLp64FnPtr, ZhpmvIlp64FnPtr, ZhpmvProvider);
define_dual_backend!(Stpmv, "stpmv", StpmvLp64FnPtr, StpmvIlp64FnPtr, StpmvProvider);
define_dual_backend!(Dtpmv, "dtpmv", DtpmvLp64FnPtr, DtpmvIlp64FnPtr, DtpmvProvider);
define_dual_backend!(Ctpmv, "ctpmv", CtpmvLp64FnPtr, CtpmvIlp64FnPtr, CtpmvProvider);
define_dual_backend!(Ztpmv, "ztpmv", ZtpmvLp64FnPtr, ZtpmvIlp64FnPtr, ZtpmvProvider);
define_dual_backend!(Stpsv, "stpsv", StpsvLp64FnPtr, StpsvIlp64FnPtr, StpsvProvider);
define_dual_backend!(Dtpsv, "dtpsv", DtpsvLp64FnPtr, DtpsvIlp64FnPtr, DtpsvProvider);
define_dual_backend!(Ctpsv, "ctpsv", CtpsvLp64FnPtr, CtpsvIlp64FnPtr, CtpsvProvider);
define_dual_backend!(Ztpsv, "ztpsv", ZtpsvLp64FnPtr, ZtpsvIlp64FnPtr, ZtpsvProvider);
define_dual_backend!(Sspr, "sspr", SsprLp64FnPtr, SsprIlp64FnPtr, SsprProvider);
define_dual_backend!(Dspr, "dspr", DsprLp64FnPtr, DsprIlp64FnPtr, DsprProvider);
define_dual_backend!(Chpr, "chpr", ChprLp64FnPtr, ChprIlp64FnPtr, ChprProvider);
define_dual_backend!(Zhpr, "zhpr", ZhprLp64FnPtr, ZhprIlp64FnPtr, ZhprProvider);
define_dual_backend!(Sspr2, "sspr2", Sspr2Lp64FnPtr, Sspr2Ilp64FnPtr, Sspr2Provider);
define_dual_backend!(Dspr2, "dspr2", Dspr2Lp64FnPtr, Dspr2Ilp64FnPtr, Dspr2Provider);
define_dual_backend!(Chpr2, "chpr2", Chpr2Lp64FnPtr, Chpr2Ilp64FnPtr, Chpr2Provider);
define_dual_backend!(Zhpr2, "zhpr2", Zhpr2Lp64FnPtr, Zhpr2Ilp64FnPtr, Zhpr2Provider);
define_dual_backend!(Ssymm, "ssymm", SsymmLp64FnPtr, SsymmIlp64FnPtr, SsymmProvider);
define_dual_backend!(Dsymm, "dsymm", DsymmLp64FnPtr, DsymmIlp64FnPtr, DsymmProvider);
define_dual_backend!(Csymm, "csymm", CsymmLp64FnPtr, CsymmIlp64FnPtr, CsymmProvider);
define_dual_backend!(Zsymm, "zsymm", ZsymmLp64FnPtr, ZsymmIlp64FnPtr, ZsymmProvider);
define_dual_backend!(Chemm, "chemm", ChemmLp64FnPtr, ChemmIlp64FnPtr, ChemmProvider);
define_dual_backend!(Zhemm, "zhemm", ZhemmLp64FnPtr, ZhemmIlp64FnPtr, ZhemmProvider);
define_dual_backend!(Ssyrk, "ssyrk", SsyrkLp64FnPtr, SsyrkIlp64FnPtr, SsyrkProvider);
define_dual_backend!(Dsyrk, "dsyrk", DsyrkLp64FnPtr, DsyrkIlp64FnPtr, DsyrkProvider);
define_dual_backend!(Csyrk, "csyrk", CsyrkLp64FnPtr, CsyrkIlp64FnPtr, CsyrkProvider);
define_dual_backend!(Zsyrk, "zsyrk", ZsyrkLp64FnPtr, ZsyrkIlp64FnPtr, ZsyrkProvider);
define_dual_backend!(Cherk, "cherk", CherkLp64FnPtr, CherkIlp64FnPtr, CherkProvider);
define_dual_backend!(Zherk, "zherk", ZherkLp64FnPtr, ZherkIlp64FnPtr, ZherkProvider);
define_dual_backend!(Ssyr2k, "ssyr2k", Ssyr2kLp64FnPtr, Ssyr2kIlp64FnPtr, Ssyr2kProvider);
define_dual_backend!(Dsyr2k, "dsyr2k", Dsyr2kLp64FnPtr, Dsyr2kIlp64FnPtr, Dsyr2kProvider);
define_dual_backend!(Csyr2k, "csyr2k", Csyr2kLp64FnPtr, Csyr2kIlp64FnPtr, Csyr2kProvider);
define_dual_backend!(Zsyr2k, "zsyr2k", Zsyr2kLp64FnPtr, Zsyr2kIlp64FnPtr, Zsyr2kProvider);
define_dual_backend!(Cher2k, "cher2k", Cher2kLp64FnPtr, Cher2kIlp64FnPtr, Cher2kProvider);
define_dual_backend!(Zher2k, "zher2k", Zher2kLp64FnPtr, Zher2kIlp64FnPtr, Zher2kProvider);
define_dual_backend!(Strmm, "strmm", StrmmLp64FnPtr, StrmmIlp64FnPtr, StrmmProvider);
define_dual_backend!(Dtrmm, "dtrmm", DtrmmLp64FnPtr, DtrmmIlp64FnPtr, DtrmmProvider);
define_dual_backend!(Ctrmm, "ctrmm", CtrmmLp64FnPtr, CtrmmIlp64FnPtr, CtrmmProvider);
define_dual_backend!(Ztrmm, "ztrmm", ZtrmmLp64FnPtr, ZtrmmIlp64FnPtr, ZtrmmProvider);
define_dual_backend!(Strsm, "strsm", StrsmLp64FnPtr, StrsmIlp64FnPtr, StrsmProvider);
define_dual_backend!(Dtrsm, "dtrsm", DtrsmLp64FnPtr, DtrsmIlp64FnPtr, DtrsmProvider);
define_dual_backend!(Ctrsm, "ctrsm", CtrsmLp64FnPtr, CtrsmIlp64FnPtr, CtrsmProvider);
define_dual_backend!(Ztrsm, "ztrsm", ZtrsmLp64FnPtr, ZtrsmIlp64FnPtr, ZtrsmProvider);

// =============================================================================
// Registration functions
// =============================================================================

fn registration_guard() -> MutexGuard<'static, ()> {
    match REGISTRATION_LOCK.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

unsafe fn register_dgemm_lp64_ptr(f: *const c_void) -> i32 {
    if f.is_null() {
        return CBLAS_INJECT_STATUS_NULL_POINTER;
    }

    let f = unsafe { std::mem::transmute::<*const c_void, DgemmLp64FnPtr>(f) };
    let _guard = registration_guard();

    #[cfg(not(feature = "ilp64"))]
    {
        if DGEMM.get().is_some() || DGEMM_LP64.get().is_some() {
            return CBLAS_INJECT_STATUS_ALREADY_REGISTERED;
        }

        if DGEMM_LP64.set(f).is_err() || DGEMM.set(f).is_err() {
            return CBLAS_INJECT_STATUS_ALREADY_REGISTERED;
        }
        CBLAS_INJECT_STATUS_OK
    }

    #[cfg(feature = "ilp64")]
    {
        match DGEMM_LP64.set(f) {
            Ok(()) => CBLAS_INJECT_STATUS_OK,
            Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
        }
    }
}

unsafe fn register_dgemm_ilp64_ptr(f: *const c_void) -> i32 {
    if f.is_null() {
        return CBLAS_INJECT_STATUS_NULL_POINTER;
    }

    let f = unsafe { std::mem::transmute::<*const c_void, DgemmIlp64FnPtr>(f) };
    let _guard = registration_guard();

    #[cfg(feature = "ilp64")]
    {
        if DGEMM.get().is_some() || DGEMM_ILP64.get().is_some() {
            return CBLAS_INJECT_STATUS_ALREADY_REGISTERED;
        }

        if DGEMM_ILP64.set(f).is_err() || DGEMM.set(f).is_err() {
            return CBLAS_INJECT_STATUS_ALREADY_REGISTERED;
        }
        CBLAS_INJECT_STATUS_OK
    }

    #[cfg(not(feature = "ilp64"))]
    {
        match DGEMM_ILP64.set(f) {
            Ok(()) => CBLAS_INJECT_STATUS_OK,
            Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
        }
    }
}

unsafe fn register_zgemm_lp64_ptr(f: *const c_void) -> i32 {
    if f.is_null() {
        return CBLAS_INJECT_STATUS_NULL_POINTER;
    }

    let f = unsafe { std::mem::transmute::<*const c_void, ZgemmLp64FnPtr>(f) };
    let _guard = registration_guard();

    #[cfg(not(feature = "ilp64"))]
    {
        if ZGEMM.get().is_some() || ZGEMM_LP64.get().is_some() {
            return CBLAS_INJECT_STATUS_ALREADY_REGISTERED;
        }

        if ZGEMM_LP64.set(f).is_err() || ZGEMM.set(f).is_err() {
            return CBLAS_INJECT_STATUS_ALREADY_REGISTERED;
        }
        CBLAS_INJECT_STATUS_OK
    }

    #[cfg(feature = "ilp64")]
    {
        match ZGEMM_LP64.set(f) {
            Ok(()) => CBLAS_INJECT_STATUS_OK,
            Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
        }
    }
}

unsafe fn register_zgemm_ilp64_ptr(f: *const c_void) -> i32 {
    if f.is_null() {
        return CBLAS_INJECT_STATUS_NULL_POINTER;
    }

    let f = unsafe { std::mem::transmute::<*const c_void, ZgemmIlp64FnPtr>(f) };
    let _guard = registration_guard();

    #[cfg(feature = "ilp64")]
    {
        if ZGEMM.get().is_some() || ZGEMM_ILP64.get().is_some() {
            return CBLAS_INJECT_STATUS_ALREADY_REGISTERED;
        }

        if ZGEMM_ILP64.set(f).is_err() || ZGEMM.set(f).is_err() {
            return CBLAS_INJECT_STATUS_ALREADY_REGISTERED;
        }
        CBLAS_INJECT_STATUS_OK
    }

    #[cfg(not(feature = "ilp64"))]
    {
        match ZGEMM_ILP64.set(f) {
            Ok(()) => CBLAS_INJECT_STATUS_OK,
            Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
        }
    }
}

/// Register an LP64 Fortran dgemm provider through the stable C API.
///
/// # Safety
///
/// `f` must be null or a valid LP64 Fortran `dgemm` function pointer that
/// remains callable for the lifetime of the process.
#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_dgemm_lp64(f: *const c_void) -> i32 {
    unsafe { register_dgemm_lp64_ptr(f) }
}

/// Register an ILP64 Fortran dgemm provider through the stable C API.
///
/// # Safety
///
/// `f` must be null or a valid ILP64 Fortran `dgemm` function pointer that
/// remains callable for the lifetime of the process.
#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_dgemm_ilp64(f: *const c_void) -> i32 {
    unsafe { register_dgemm_ilp64_ptr(f) }
}

/// Register an LP64 Fortran zgemm provider through the stable C API.
///
/// # Safety
///
/// `f` must be null or a valid LP64 Fortran `zgemm` function pointer that
/// remains callable for the lifetime of the process.
#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_zgemm_lp64(f: *const c_void) -> i32 {
    unsafe { register_zgemm_lp64_ptr(f) }
}

/// Register an ILP64 Fortran zgemm provider through the stable C API.
///
/// # Safety
///
/// `f` must be null or a valid ILP64 Fortran `zgemm` function pointer that
/// remains callable for the lifetime of the process.
#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_zgemm_ilp64(f: *const c_void) -> i32 {
    unsafe { register_zgemm_ilp64_ptr(f) }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_sgemm_lp64(f: *const c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    let f: SgemmLp64FnPtr = unsafe { std::mem::transmute(f) };
    match SGEMM_LP64.set(f) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_sgemm_ilp64(f: *const c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    let f: SgemmIlp64FnPtr = unsafe { std::mem::transmute(f) };
    match SGEMM_ILP64.set(f) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_cgemm_lp64(f: *const c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    let f: CgemmLp64FnPtr = unsafe { std::mem::transmute(f) };
    match CGEMM_LP64.set(f) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_cgemm_ilp64(f: *const c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    let f: CgemmIlp64FnPtr = unsafe { std::mem::transmute(f) };
    match CGEMM_ILP64.set(f) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

/// Return the BLAS integer width used by unprefixed CBLAS symbols in this build.
#[no_mangle]
pub extern "C" fn cblas_inject_blas_int_width() -> i32 {
    // Transitional behavior: the `ilp64` Cargo feature still changes the
    // unprefixed `cblas_*` ABI, so report the actual loaded build.
    (std::mem::size_of::<blasint>() * 8) as i32
}

/// Return whether this build accepts LP64 explicit provider registration.
#[no_mangle]
pub extern "C" fn cblas_inject_supports_lp64_registration() -> i32 {
    1
}

/// Return whether this build accepts ILP64 explicit provider registration.
#[no_mangle]
pub extern "C" fn cblas_inject_supports_ilp64_registration() -> i32 {
    1
}

/// Compatibility alias for LP64 explicit provider registration support.
#[no_mangle]
pub extern "C" fn cblas_inject_supports_lp64() -> i32 {
    cblas_inject_supports_lp64_registration()
}

/// Compatibility alias for ILP64 explicit provider registration support.
#[no_mangle]
pub extern "C" fn cblas_inject_supports_ilp64() -> i32 {
    cblas_inject_supports_ilp64_registration()
}

// BLAS Level 1-3 registration
// (Legacy LP64 registration aliases)

#[no_mangle]
pub unsafe extern "C" fn register_sswap(f: SswapFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSWAP>].set(f) };
    let _ = paste::paste! { [<Sswap_LP64>].set(std::mem::transmute::<SswapFnPtr, SswapLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dswap(f: DswapFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSWAP>].set(f) };
    let _ = paste::paste! { [<Dswap_LP64>].set(std::mem::transmute::<DswapFnPtr, DswapLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_cswap(f: CswapFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CSWAP>].set(f) };
    let _ = paste::paste! { [<Cswap_LP64>].set(std::mem::transmute::<CswapFnPtr, CswapLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zswap(f: ZswapFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZSWAP>].set(f) };
    let _ = paste::paste! { [<Zswap_LP64>].set(std::mem::transmute::<ZswapFnPtr, ZswapLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_scopy(f: ScopyFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SCOPY>].set(f) };
    let _ = paste::paste! { [<Scopy_LP64>].set(std::mem::transmute::<ScopyFnPtr, ScopyLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dcopy(f: DcopyFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DCOPY>].set(f) };
    let _ = paste::paste! { [<Dcopy_LP64>].set(std::mem::transmute::<DcopyFnPtr, DcopyLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ccopy(f: CcopyFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CCOPY>].set(f) };
    let _ = paste::paste! { [<Ccopy_LP64>].set(std::mem::transmute::<CcopyFnPtr, CcopyLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zcopy(f: ZcopyFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZCOPY>].set(f) };
    let _ = paste::paste! { [<Zcopy_LP64>].set(std::mem::transmute::<ZcopyFnPtr, ZcopyLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_saxpy(f: SaxpyFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SAXPY>].set(f) };
    let _ = paste::paste! { [<Saxpy_LP64>].set(std::mem::transmute::<SaxpyFnPtr, SaxpyLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_daxpy(f: DaxpyFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DAXPY>].set(f) };
    let _ = paste::paste! { [<Daxpy_LP64>].set(std::mem::transmute::<DaxpyFnPtr, DaxpyLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_caxpy(f: CaxpyFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CAXPY>].set(f) };
    let _ = paste::paste! { [<Caxpy_LP64>].set(std::mem::transmute::<CaxpyFnPtr, CaxpyLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zaxpy(f: ZaxpyFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZAXPY>].set(f) };
    let _ = paste::paste! { [<Zaxpy_LP64>].set(std::mem::transmute::<ZaxpyFnPtr, ZaxpyLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_sscal(f: SscalFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSCAL>].set(f) };
    let _ = paste::paste! { [<Sscal_LP64>].set(std::mem::transmute::<SscalFnPtr, SscalLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dscal(f: DscalFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSCAL>].set(f) };
    let _ = paste::paste! { [<Dscal_LP64>].set(std::mem::transmute::<DscalFnPtr, DscalLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_cscal(f: CscalFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CSCAL>].set(f) };
    let _ = paste::paste! { [<Cscal_LP64>].set(std::mem::transmute::<CscalFnPtr, CscalLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zscal(f: ZscalFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZSCAL>].set(f) };
    let _ = paste::paste! { [<Zscal_LP64>].set(std::mem::transmute::<ZscalFnPtr, ZscalLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_csscal(f: CsscalFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CSSCAL>].set(f) };
    let _ = paste::paste! { [<Csscal_LP64>].set(std::mem::transmute::<CsscalFnPtr, CsscalLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zdscal(f: ZdscalFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZDSCAL>].set(f) };
    let _ = paste::paste! { [<Zdscal_LP64>].set(std::mem::transmute::<ZdscalFnPtr, ZdscalLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_srot(f: SrotFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SROT>].set(f) };
    let _ = paste::paste! { [<Srot_LP64>].set(std::mem::transmute::<SrotFnPtr, SrotLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_drot(f: DrotFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DROT>].set(f) };
    let _ = paste::paste! { [<Drot_LP64>].set(std::mem::transmute::<DrotFnPtr, DrotLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_srotg(f: SrotgFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SROTG>].set(f) };
    let _ = paste::paste! { [<Srotg_LP64>].set(std::mem::transmute::<SrotgFnPtr, SrotgLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_drotg(f: DrotgFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DROTG>].set(f) };
    let _ = paste::paste! { [<Drotg_LP64>].set(std::mem::transmute::<DrotgFnPtr, DrotgLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_srotm(f: SrotmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SROTM>].set(f) };
    let _ = paste::paste! { [<Srotm_LP64>].set(std::mem::transmute::<SrotmFnPtr, SrotmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_drotm(f: DrotmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DROTM>].set(f) };
    let _ = paste::paste! { [<Drotm_LP64>].set(std::mem::transmute::<DrotmFnPtr, DrotmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_srotmg(f: SrotmgFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SROTMG>].set(f) };
    let _ = paste::paste! { [<Srotmg_LP64>].set(std::mem::transmute::<SrotmgFnPtr, SrotmgLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_drotmg(f: DrotmgFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DROTMG>].set(f) };
    let _ = paste::paste! { [<Drotmg_LP64>].set(std::mem::transmute::<DrotmgFnPtr, DrotmgLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_scabs1(f: Scabs1FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SCABS1>].set(f) };
    let _ = paste::paste! { [<Scabs1_LP64>].set(std::mem::transmute::<Scabs1FnPtr, Scabs1Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dcabs1(f: Dcabs1FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DCABS1>].set(f) };
    let _ = paste::paste! { [<Dcabs1_LP64>].set(std::mem::transmute::<Dcabs1FnPtr, Dcabs1Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_sdot(f: SdotFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SDOT>].set(f) };
    let _ = paste::paste! { [<Sdot_LP64>].set(std::mem::transmute::<SdotFnPtr, SdotLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ddot(f: DdotFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DDOT>].set(f) };
    let _ = paste::paste! { [<Ddot_LP64>].set(std::mem::transmute::<DdotFnPtr, DdotLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_sdsdot(f: SdsdotFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SDSDOT>].set(f) };
    let _ = paste::paste! { [<Sdsdot_LP64>].set(std::mem::transmute::<SdsdotFnPtr, SdsdotLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dsdot(f: DsdotFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSDOT>].set(f) };
    let _ = paste::paste! { [<Dsdot_LP64>].set(std::mem::transmute::<DsdotFnPtr, DsdotLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_snrm2(f: Snrm2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SNRM2>].set(f) };
    let _ = paste::paste! { [<Snrm2_LP64>].set(std::mem::transmute::<Snrm2FnPtr, Snrm2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dnrm2(f: Dnrm2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DNRM2>].set(f) };
    let _ = paste::paste! { [<Dnrm2_LP64>].set(std::mem::transmute::<Dnrm2FnPtr, Dnrm2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_scnrm2(f: Scnrm2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SCNRM2>].set(f) };
    let _ = paste::paste! { [<Scnrm2_LP64>].set(std::mem::transmute::<Scnrm2FnPtr, Scnrm2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dznrm2(f: Dznrm2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DZNRM2>].set(f) };
    let _ = paste::paste! { [<Dznrm2_LP64>].set(std::mem::transmute::<Dznrm2FnPtr, Dznrm2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_sasum(f: SasumFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SASUM>].set(f) };
    let _ = paste::paste! { [<Sasum_LP64>].set(std::mem::transmute::<SasumFnPtr, SasumLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dasum(f: DasumFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DASUM>].set(f) };
    let _ = paste::paste! { [<Dasum_LP64>].set(std::mem::transmute::<DasumFnPtr, DasumLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_scasum(f: ScasumFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SCASUM>].set(f) };
    let _ = paste::paste! { [<Scasum_LP64>].set(std::mem::transmute::<ScasumFnPtr, ScasumLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dzasum(f: DzasumFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DZASUM>].set(f) };
    let _ = paste::paste! { [<Dzasum_LP64>].set(std::mem::transmute::<DzasumFnPtr, DzasumLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_isamax(f: IsamaxFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ISAMAX>].set(f) };
    let _ = paste::paste! { [<Isamax_LP64>].set(std::mem::transmute::<IsamaxFnPtr, IsamaxLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_idamax(f: IdamaxFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<IDAMAX>].set(f) };
    let _ = paste::paste! { [<Idamax_LP64>].set(std::mem::transmute::<IdamaxFnPtr, IdamaxLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_icamax(f: IcamaxFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ICAMAX>].set(f) };
    let _ = paste::paste! { [<Icamax_LP64>].set(std::mem::transmute::<IcamaxFnPtr, IcamaxLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_izamax(f: IzamaxFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<IZAMAX>].set(f) };
    let _ = paste::paste! { [<Izamax_LP64>].set(std::mem::transmute::<IzamaxFnPtr, IzamaxLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_sgemv(f: SgemvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SGEMV>].set(f) };
    let _ = paste::paste! { [<Sgemv_LP64>].set(std::mem::transmute::<SgemvFnPtr, SgemvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dgemv(f: DgemvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DGEMV>].set(f) };
    let _ = paste::paste! { [<Dgemv_LP64>].set(std::mem::transmute::<DgemvFnPtr, DgemvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_cgemv(f: CgemvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CGEMV>].set(f) };
    let _ = paste::paste! { [<Cgemv_LP64>].set(std::mem::transmute::<CgemvFnPtr, CgemvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zgemv(f: ZgemvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZGEMV>].set(f) };
    let _ = paste::paste! { [<Zgemv_LP64>].set(std::mem::transmute::<ZgemvFnPtr, ZgemvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_sgbmv(f: SgbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SGBMV>].set(f) };
    let _ = paste::paste! { [<Sgbmv_LP64>].set(std::mem::transmute::<SgbmvFnPtr, SgbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dgbmv(f: DgbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DGBMV>].set(f) };
    let _ = paste::paste! { [<Dgbmv_LP64>].set(std::mem::transmute::<DgbmvFnPtr, DgbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_cgbmv(f: CgbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CGBMV>].set(f) };
    let _ = paste::paste! { [<Cgbmv_LP64>].set(std::mem::transmute::<CgbmvFnPtr, CgbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zgbmv(f: ZgbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZGBMV>].set(f) };
    let _ = paste::paste! { [<Zgbmv_LP64>].set(std::mem::transmute::<ZgbmvFnPtr, ZgbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ssymv(f: SsymvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSYMV>].set(f) };
    let _ = paste::paste! { [<Ssymv_LP64>].set(std::mem::transmute::<SsymvFnPtr, SsymvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dsymv(f: DsymvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSYMV>].set(f) };
    let _ = paste::paste! { [<Dsymv_LP64>].set(std::mem::transmute::<DsymvFnPtr, DsymvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_chemv(f: ChemvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CHEMV>].set(f) };
    let _ = paste::paste! { [<Chemv_LP64>].set(std::mem::transmute::<ChemvFnPtr, ChemvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zhemv(f: ZhemvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZHEMV>].set(f) };
    let _ = paste::paste! { [<Zhemv_LP64>].set(std::mem::transmute::<ZhemvFnPtr, ZhemvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ssbmv(f: SsbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSBMV>].set(f) };
    let _ = paste::paste! { [<Ssbmv_LP64>].set(std::mem::transmute::<SsbmvFnPtr, SsbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dsbmv(f: DsbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSBMV>].set(f) };
    let _ = paste::paste! { [<Dsbmv_LP64>].set(std::mem::transmute::<DsbmvFnPtr, DsbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_chbmv(f: ChbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CHBMV>].set(f) };
    let _ = paste::paste! { [<Chbmv_LP64>].set(std::mem::transmute::<ChbmvFnPtr, ChbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zhbmv(f: ZhbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZHBMV>].set(f) };
    let _ = paste::paste! { [<Zhbmv_LP64>].set(std::mem::transmute::<ZhbmvFnPtr, ZhbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_strmv(f: StrmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<STRMV>].set(f) };
    let _ = paste::paste! { [<Strmv_LP64>].set(std::mem::transmute::<StrmvFnPtr, StrmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dtrmv(f: DtrmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DTRMV>].set(f) };
    let _ = paste::paste! { [<Dtrmv_LP64>].set(std::mem::transmute::<DtrmvFnPtr, DtrmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ctrmv(f: CtrmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CTRMV>].set(f) };
    let _ = paste::paste! { [<Ctrmv_LP64>].set(std::mem::transmute::<CtrmvFnPtr, CtrmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ztrmv(f: ZtrmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZTRMV>].set(f) };
    let _ = paste::paste! { [<Ztrmv_LP64>].set(std::mem::transmute::<ZtrmvFnPtr, ZtrmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_strsv(f: StrsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<STRSV>].set(f) };
    let _ = paste::paste! { [<Strsv_LP64>].set(std::mem::transmute::<StrsvFnPtr, StrsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dtrsv(f: DtrsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DTRSV>].set(f) };
    let _ = paste::paste! { [<Dtrsv_LP64>].set(std::mem::transmute::<DtrsvFnPtr, DtrsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ctrsv(f: CtrsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CTRSV>].set(f) };
    let _ = paste::paste! { [<Ctrsv_LP64>].set(std::mem::transmute::<CtrsvFnPtr, CtrsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ztrsv(f: ZtrsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZTRSV>].set(f) };
    let _ = paste::paste! { [<Ztrsv_LP64>].set(std::mem::transmute::<ZtrsvFnPtr, ZtrsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_stbmv(f: StbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<STBMV>].set(f) };
    let _ = paste::paste! { [<Stbmv_LP64>].set(std::mem::transmute::<StbmvFnPtr, StbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dtbmv(f: DtbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DTBMV>].set(f) };
    let _ = paste::paste! { [<Dtbmv_LP64>].set(std::mem::transmute::<DtbmvFnPtr, DtbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ctbmv(f: CtbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CTBMV>].set(f) };
    let _ = paste::paste! { [<Ctbmv_LP64>].set(std::mem::transmute::<CtbmvFnPtr, CtbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ztbmv(f: ZtbmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZTBMV>].set(f) };
    let _ = paste::paste! { [<Ztbmv_LP64>].set(std::mem::transmute::<ZtbmvFnPtr, ZtbmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_stbsv(f: StbsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<STBSV>].set(f) };
    let _ = paste::paste! { [<Stbsv_LP64>].set(std::mem::transmute::<StbsvFnPtr, StbsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dtbsv(f: DtbsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DTBSV>].set(f) };
    let _ = paste::paste! { [<Dtbsv_LP64>].set(std::mem::transmute::<DtbsvFnPtr, DtbsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ctbsv(f: CtbsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CTBSV>].set(f) };
    let _ = paste::paste! { [<Ctbsv_LP64>].set(std::mem::transmute::<CtbsvFnPtr, CtbsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ztbsv(f: ZtbsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZTBSV>].set(f) };
    let _ = paste::paste! { [<Ztbsv_LP64>].set(std::mem::transmute::<ZtbsvFnPtr, ZtbsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_sger(f: SgerFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SGER>].set(f) };
    let _ = paste::paste! { [<Sger_LP64>].set(std::mem::transmute::<SgerFnPtr, SgerLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dger(f: DgerFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DGER>].set(f) };
    let _ = paste::paste! { [<Dger_LP64>].set(std::mem::transmute::<DgerFnPtr, DgerLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_cgeru(f: CgeruFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CGERU>].set(f) };
    let _ = paste::paste! { [<Cgeru_LP64>].set(std::mem::transmute::<CgeruFnPtr, CgeruLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_cgerc(f: CgercFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CGERC>].set(f) };
    let _ = paste::paste! { [<Cgerc_LP64>].set(std::mem::transmute::<CgercFnPtr, CgercLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zgeru(f: ZgeruFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZGERU>].set(f) };
    let _ = paste::paste! { [<Zgeru_LP64>].set(std::mem::transmute::<ZgeruFnPtr, ZgeruLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zgerc(f: ZgercFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZGERC>].set(f) };
    let _ = paste::paste! { [<Zgerc_LP64>].set(std::mem::transmute::<ZgercFnPtr, ZgercLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ssyr(f: SsyrFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSYR>].set(f) };
    let _ = paste::paste! { [<Ssyr_LP64>].set(std::mem::transmute::<SsyrFnPtr, SsyrLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dsyr(f: DsyrFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSYR>].set(f) };
    let _ = paste::paste! { [<Dsyr_LP64>].set(std::mem::transmute::<DsyrFnPtr, DsyrLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_cher(f: CherFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CHER>].set(f) };
    let _ = paste::paste! { [<Cher_LP64>].set(std::mem::transmute::<CherFnPtr, CherLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zher(f: ZherFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZHER>].set(f) };
    let _ = paste::paste! { [<Zher_LP64>].set(std::mem::transmute::<ZherFnPtr, ZherLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ssyr2(f: Ssyr2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSYR2>].set(f) };
    let _ = paste::paste! { [<Ssyr2_LP64>].set(std::mem::transmute::<Ssyr2FnPtr, Ssyr2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dsyr2(f: Dsyr2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSYR2>].set(f) };
    let _ = paste::paste! { [<Dsyr2_LP64>].set(std::mem::transmute::<Dsyr2FnPtr, Dsyr2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_cher2(f: Cher2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CHER2>].set(f) };
    let _ = paste::paste! { [<Cher2_LP64>].set(std::mem::transmute::<Cher2FnPtr, Cher2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zher2(f: Zher2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZHER2>].set(f) };
    let _ = paste::paste! { [<Zher2_LP64>].set(std::mem::transmute::<Zher2FnPtr, Zher2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_sspmv(f: SspmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSPMV>].set(f) };
    let _ = paste::paste! { [<Sspmv_LP64>].set(std::mem::transmute::<SspmvFnPtr, SspmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dspmv(f: DspmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSPMV>].set(f) };
    let _ = paste::paste! { [<Dspmv_LP64>].set(std::mem::transmute::<DspmvFnPtr, DspmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_chpmv(f: ChpmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CHPMV>].set(f) };
    let _ = paste::paste! { [<Chpmv_LP64>].set(std::mem::transmute::<ChpmvFnPtr, ChpmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zhpmv(f: ZhpmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZHPMV>].set(f) };
    let _ = paste::paste! { [<Zhpmv_LP64>].set(std::mem::transmute::<ZhpmvFnPtr, ZhpmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_stpmv(f: StpmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<STPMV>].set(f) };
    let _ = paste::paste! { [<Stpmv_LP64>].set(std::mem::transmute::<StpmvFnPtr, StpmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dtpmv(f: DtpmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DTPMV>].set(f) };
    let _ = paste::paste! { [<Dtpmv_LP64>].set(std::mem::transmute::<DtpmvFnPtr, DtpmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ctpmv(f: CtpmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CTPMV>].set(f) };
    let _ = paste::paste! { [<Ctpmv_LP64>].set(std::mem::transmute::<CtpmvFnPtr, CtpmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ztpmv(f: ZtpmvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZTPMV>].set(f) };
    let _ = paste::paste! { [<Ztpmv_LP64>].set(std::mem::transmute::<ZtpmvFnPtr, ZtpmvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_stpsv(f: StpsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<STPSV>].set(f) };
    let _ = paste::paste! { [<Stpsv_LP64>].set(std::mem::transmute::<StpsvFnPtr, StpsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dtpsv(f: DtpsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DTPSV>].set(f) };
    let _ = paste::paste! { [<Dtpsv_LP64>].set(std::mem::transmute::<DtpsvFnPtr, DtpsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ctpsv(f: CtpsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CTPSV>].set(f) };
    let _ = paste::paste! { [<Ctpsv_LP64>].set(std::mem::transmute::<CtpsvFnPtr, CtpsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ztpsv(f: ZtpsvFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZTPSV>].set(f) };
    let _ = paste::paste! { [<Ztpsv_LP64>].set(std::mem::transmute::<ZtpsvFnPtr, ZtpsvLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_sspr(f: SsprFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSPR>].set(f) };
    let _ = paste::paste! { [<Sspr_LP64>].set(std::mem::transmute::<SsprFnPtr, SsprLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dspr(f: DsprFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSPR>].set(f) };
    let _ = paste::paste! { [<Dspr_LP64>].set(std::mem::transmute::<DsprFnPtr, DsprLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_chpr(f: ChprFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CHPR>].set(f) };
    let _ = paste::paste! { [<Chpr_LP64>].set(std::mem::transmute::<ChprFnPtr, ChprLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zhpr(f: ZhprFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZHPR>].set(f) };
    let _ = paste::paste! { [<Zhpr_LP64>].set(std::mem::transmute::<ZhprFnPtr, ZhprLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_sspr2(f: Sspr2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSPR2>].set(f) };
    let _ = paste::paste! { [<Sspr2_LP64>].set(std::mem::transmute::<Sspr2FnPtr, Sspr2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dspr2(f: Dspr2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSPR2>].set(f) };
    let _ = paste::paste! { [<Dspr2_LP64>].set(std::mem::transmute::<Dspr2FnPtr, Dspr2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_chpr2(f: Chpr2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CHPR2>].set(f) };
    let _ = paste::paste! { [<Chpr2_LP64>].set(std::mem::transmute::<Chpr2FnPtr, Chpr2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zhpr2(f: Zhpr2FnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZHPR2>].set(f) };
    let _ = paste::paste! { [<Zhpr2_LP64>].set(std::mem::transmute::<Zhpr2FnPtr, Zhpr2Lp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ssymm(f: SsymmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSYMM>].set(f) };
    let _ = paste::paste! { [<Ssymm_LP64>].set(std::mem::transmute::<SsymmFnPtr, SsymmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dsymm(f: DsymmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSYMM>].set(f) };
    let _ = paste::paste! { [<Dsymm_LP64>].set(std::mem::transmute::<DsymmFnPtr, DsymmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_csymm(f: CsymmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CSYMM>].set(f) };
    let _ = paste::paste! { [<Csymm_LP64>].set(std::mem::transmute::<CsymmFnPtr, CsymmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zsymm(f: ZsymmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZSYMM>].set(f) };
    let _ = paste::paste! { [<Zsymm_LP64>].set(std::mem::transmute::<ZsymmFnPtr, ZsymmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_chemm(f: ChemmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CHEMM>].set(f) };
    let _ = paste::paste! { [<Chemm_LP64>].set(std::mem::transmute::<ChemmFnPtr, ChemmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zhemm(f: ZhemmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZHEMM>].set(f) };
    let _ = paste::paste! { [<Zhemm_LP64>].set(std::mem::transmute::<ZhemmFnPtr, ZhemmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ssyrk(f: SsyrkFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSYRK>].set(f) };
    let _ = paste::paste! { [<Ssyrk_LP64>].set(std::mem::transmute::<SsyrkFnPtr, SsyrkLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dsyrk(f: DsyrkFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSYRK>].set(f) };
    let _ = paste::paste! { [<Dsyrk_LP64>].set(std::mem::transmute::<DsyrkFnPtr, DsyrkLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_csyrk(f: CsyrkFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CSYRK>].set(f) };
    let _ = paste::paste! { [<Csyrk_LP64>].set(std::mem::transmute::<CsyrkFnPtr, CsyrkLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zsyrk(f: ZsyrkFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZSYRK>].set(f) };
    let _ = paste::paste! { [<Zsyrk_LP64>].set(std::mem::transmute::<ZsyrkFnPtr, ZsyrkLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_cherk(f: CherkFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CHERK>].set(f) };
    let _ = paste::paste! { [<Cherk_LP64>].set(std::mem::transmute::<CherkFnPtr, CherkLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zherk(f: ZherkFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZHERK>].set(f) };
    let _ = paste::paste! { [<Zherk_LP64>].set(std::mem::transmute::<ZherkFnPtr, ZherkLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ssyr2k(f: Ssyr2kFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<SSYR2K>].set(f) };
    let _ = paste::paste! { [<Ssyr2k_LP64>].set(std::mem::transmute::<Ssyr2kFnPtr, Ssyr2kLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dsyr2k(f: Dsyr2kFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DSYR2K>].set(f) };
    let _ = paste::paste! { [<Dsyr2k_LP64>].set(std::mem::transmute::<Dsyr2kFnPtr, Dsyr2kLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_csyr2k(f: Csyr2kFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CSYR2K>].set(f) };
    let _ = paste::paste! { [<Csyr2k_LP64>].set(std::mem::transmute::<Csyr2kFnPtr, Csyr2kLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zsyr2k(f: Zsyr2kFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZSYR2K>].set(f) };
    let _ = paste::paste! { [<Zsyr2k_LP64>].set(std::mem::transmute::<Zsyr2kFnPtr, Zsyr2kLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_cher2k(f: Cher2kFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CHER2K>].set(f) };
    let _ = paste::paste! { [<Cher2k_LP64>].set(std::mem::transmute::<Cher2kFnPtr, Cher2kLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_zher2k(f: Zher2kFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZHER2K>].set(f) };
    let _ = paste::paste! { [<Zher2k_LP64>].set(std::mem::transmute::<Zher2kFnPtr, Zher2kLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_strmm(f: StrmmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<STRMM>].set(f) };
    let _ = paste::paste! { [<Strmm_LP64>].set(std::mem::transmute::<StrmmFnPtr, StrmmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dtrmm(f: DtrmmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DTRMM>].set(f) };
    let _ = paste::paste! { [<Dtrmm_LP64>].set(std::mem::transmute::<DtrmmFnPtr, DtrmmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ctrmm(f: CtrmmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CTRMM>].set(f) };
    let _ = paste::paste! { [<Ctrmm_LP64>].set(std::mem::transmute::<CtrmmFnPtr, CtrmmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ztrmm(f: ZtrmmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZTRMM>].set(f) };
    let _ = paste::paste! { [<Ztrmm_LP64>].set(std::mem::transmute::<ZtrmmFnPtr, ZtrmmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_strsm(f: StrsmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<STRSM>].set(f) };
    let _ = paste::paste! { [<Strsm_LP64>].set(std::mem::transmute::<StrsmFnPtr, StrsmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_dtrsm(f: DtrsmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<DTRSM>].set(f) };
    let _ = paste::paste! { [<Dtrsm_LP64>].set(std::mem::transmute::<DtrsmFnPtr, DtrsmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ctrsm(f: CtrsmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<CTRSM>].set(f) };
    let _ = paste::paste! { [<Ctrsm_LP64>].set(std::mem::transmute::<CtrsmFnPtr, CtrsmLp64FnPtr>(f)) };
}

#[no_mangle]
pub unsafe extern "C" fn register_ztrsm(f: ZtrsmFnPtr) {
    // Legacy LP64 registration - populate both old static and new LP64 storage
    let _ = paste::paste! { [<ZTRSM>].set(f) };
    let _ = paste::paste! { [<Ztrsm_LP64>].set(std::mem::transmute::<ZtrsmFnPtr, ZtrsmLp64FnPtr>(f)) };
}


// BLAS Level 3 registration (gemm)

#[no_mangle]
pub unsafe extern "C" fn register_dgemm(f: DgemmFnPtr) {
    let _guard = registration_guard();
    #[cfg(not(feature = "ilp64"))]
    if DGEMM.get().is_some() || DGEMM_LP64.get().is_some() {
        panic!("dgemm already registered (can only be set once)");
    }
    #[cfg(feature = "ilp64")]
    if DGEMM.get().is_some() || DGEMM_ILP64.get().is_some() {
        panic!("dgemm already registered (can only be set once)");
    }

    #[cfg(not(feature = "ilp64"))]
    {
        DGEMM_LP64
            .set(f)
            .expect("dgemm already registered (can only be set once)");
    }
    #[cfg(feature = "ilp64")]
    {
        DGEMM_ILP64
            .set(f)
            .expect("dgemm already registered (can only be set once)");
    }
    DGEMM
        .set(f)
        .expect("dgemm already registered (can only be set once)");
}

#[no_mangle]
pub unsafe extern "C" fn register_sgemm(f: SgemmFnPtr) {
    SGEMM
        .set(f)
        .expect("sgemm already registered (can only be set once)");
    #[cfg(not(feature = "ilp64"))]
    {
        let _ = SGEMM_LP64.set(f);
    }
}

#[no_mangle]
pub unsafe extern "C" fn register_zgemm(f: ZgemmFnPtr) {
    let _guard = registration_guard();
    #[cfg(not(feature = "ilp64"))]
    if ZGEMM.get().is_some() || ZGEMM_LP64.get().is_some() {
        panic!("zgemm already registered (can only be set once)");
    }
    #[cfg(feature = "ilp64")]
    if ZGEMM.get().is_some() || ZGEMM_ILP64.get().is_some() {
        panic!("zgemm already registered (can only be set once)");
    }

    #[cfg(not(feature = "ilp64"))]
    {
        ZGEMM_LP64
            .set(f)
            .expect("zgemm already registered (can only be set once)");
    }
    #[cfg(feature = "ilp64")]
    {
        ZGEMM_ILP64
            .set(f)
            .expect("zgemm already registered (can only be set once)");
    }
    ZGEMM
        .set(f)
        .expect("zgemm already registered (can only be set once)");
}

#[no_mangle]
pub unsafe extern "C" fn register_cgemm(f: CgemmFnPtr) {
    CGEMM
        .set(f)
        .expect("cgemm already registered (can only be set once)");
    #[cfg(not(feature = "ilp64"))]
    {
        let _ = CGEMM_LP64.set(f);
    }
}

/// Register the Fortran cdotu function pointer (return value convention, LP64).
///
/// # Safety
///
/// The function pointer must be a valid Fortran cdotu implementation
/// using the return value convention, accepting i32 blasint parameters.
#[no_mangle]
pub unsafe extern "C" fn register_cdotu(f: CdotuFnPtr) {
    CDOTU_LP64_PTR
        .set(FnPtrWrapper(f as *const ()))
        .expect("cdotu already registered (can only be set once)");
}

/// Register a raw cdotu function pointer (LP64).
///
/// # Safety
///
/// The function pointer must be a valid Fortran cdotu implementation
/// accepting i32 blasint parameters.
#[no_mangle]
pub unsafe extern "C" fn register_cdotu_raw(ptr: *const ()) {
    CDOTU_LP64_PTR
        .set(FnPtrWrapper(ptr))
        .expect("cdotu already registered (can only be set once)");
}

/// Register the Fortran zdotu function pointer (return value convention, LP64).
///
/// # Safety
///
/// The function pointer must be a valid Fortran zdotu implementation
/// using the return value convention, accepting i32 blasint parameters.
#[no_mangle]
pub unsafe extern "C" fn register_zdotu(f: ZdotuFnPtr) {
    ZDOTU_LP64_PTR
        .set(FnPtrWrapper(f as *const ()))
        .expect("zdotu already registered (can only be set once)");
}

/// Register a raw zdotu function pointer (LP64).
///
/// # Safety
///
/// The function pointer must be a valid Fortran zdotu implementation
/// accepting i32 blasint parameters.
#[no_mangle]
pub unsafe extern "C" fn register_zdotu_raw(ptr: *const ()) {
    ZDOTU_LP64_PTR
        .set(FnPtrWrapper(ptr))
        .expect("zdotu already registered (can only be set once)");
}

/// Register the Fortran cdotc function pointer (return value convention, LP64).
///
/// # Safety
///
/// The function pointer must be a valid Fortran cdotc implementation
/// using the return value convention, accepting i32 blasint parameters.
#[no_mangle]
pub unsafe extern "C" fn register_cdotc(f: CdotcFnPtr) {
    CDOTC_LP64_PTR
        .set(FnPtrWrapper(f as *const ()))
        .expect("cdotc already registered (can only be set once)");
}

/// Register a raw cdotc function pointer (LP64).
///
/// # Safety
///
/// The function pointer must be a valid Fortran cdotc implementation
/// accepting i32 blasint parameters.
#[no_mangle]
pub unsafe extern "C" fn register_cdotc_raw(ptr: *const ()) {
    CDOTC_LP64_PTR
        .set(FnPtrWrapper(ptr))
        .expect("cdotc already registered (can only be set once)");
}

/// Register the Fortran zdotc function pointer (return value convention, LP64).
///
/// # Safety
///
/// The function pointer must be a valid Fortran zdotc implementation
/// using the return value convention, accepting i32 blasint parameters.
#[no_mangle]
pub unsafe extern "C" fn register_zdotc(f: ZdotcFnPtr) {
    ZDOTC_LP64_PTR
        .set(FnPtrWrapper(f as *const ()))
        .expect("zdotc already registered (can only be set once)");
}

/// Register a raw zdotc function pointer (LP64).
///
/// # Safety
///
/// The function pointer must be a valid Fortran zdotc implementation
/// accepting i32 blasint parameters.
#[no_mangle]
pub unsafe extern "C" fn register_zdotc_raw(ptr: *const ()) {
    ZDOTC_LP64_PTR
        .set(FnPtrWrapper(ptr))
        .expect("zdotc already registered (can only be set once)");
}

// Complex dot LP64/ILP64 raw-pointer C API registration

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_cdotu_lp64(f: *const std::ffi::c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    match CDOTU_LP64_PTR.set(FnPtrWrapper(f as *const ())) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_cdotu_ilp64(f: *const std::ffi::c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    match CDOTU_ILP64_PTR.set(FnPtrWrapper(f as *const ())) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_zdotu_lp64(f: *const std::ffi::c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    match ZDOTU_LP64_PTR.set(FnPtrWrapper(f as *const ())) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_zdotu_ilp64(f: *const std::ffi::c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    match ZDOTU_ILP64_PTR.set(FnPtrWrapper(f as *const ())) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_cdotc_lp64(f: *const std::ffi::c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    match CDOTC_LP64_PTR.set(FnPtrWrapper(f as *const ())) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_cdotc_ilp64(f: *const std::ffi::c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    match CDOTC_ILP64_PTR.set(FnPtrWrapper(f as *const ())) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_zdotc_lp64(f: *const std::ffi::c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    match ZDOTC_LP64_PTR.set(FnPtrWrapper(f as *const ())) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub unsafe extern "C" fn cblas_inject_register_zdotc_ilp64(f: *const std::ffi::c_void) -> i32 {
    if f.is_null() { return CBLAS_INJECT_STATUS_NULL_POINTER; }
    match ZDOTC_ILP64_PTR.set(FnPtrWrapper(f as *const ())) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

// Complex dot getters (LP64-first with ILP64 fallback)

pub(crate) fn get_cdotu_lp64_ptr() -> *const () {
    if let Some(p) = CDOTU_LP64_PTR.get() {
        return p.0;
    }
    CDOTU_ILP64_PTR.get().map(|p| p.0).expect("cdotu not registered")
}

pub(crate) fn get_cdotu_ilp64_ptr() -> *const () {
    if let Some(p) = CDOTU_ILP64_PTR.get() {
        return p.0;
    }
    CDOTU_LP64_PTR.get().map(|p| p.0).expect("cdotu not registered")
}

pub(crate) fn get_zdotu_lp64_ptr() -> *const () {
    if let Some(p) = ZDOTU_LP64_PTR.get() {
        return p.0;
    }
    ZDOTU_ILP64_PTR.get().map(|p| p.0).expect("zdotu not registered")
}

pub(crate) fn get_zdotu_ilp64_ptr() -> *const () {
    if let Some(p) = ZDOTU_ILP64_PTR.get() {
        return p.0;
    }
    ZDOTU_LP64_PTR.get().map(|p| p.0).expect("zdotu not registered")
}

pub(crate) fn get_cdotc_lp64_ptr() -> *const () {
    if let Some(p) = CDOTC_LP64_PTR.get() {
        return p.0;
    }
    CDOTC_ILP64_PTR.get().map(|p| p.0).expect("cdotc not registered")
}

pub(crate) fn get_cdotc_ilp64_ptr() -> *const () {
    if let Some(p) = CDOTC_ILP64_PTR.get() {
        return p.0;
    }
    CDOTC_LP64_PTR.get().map(|p| p.0).expect("cdotc not registered")
}

pub(crate) fn get_zdotc_lp64_ptr() -> *const () {
    if let Some(p) = ZDOTC_LP64_PTR.get() {
        return p.0;
    }
    ZDOTC_ILP64_PTR.get().map(|p| p.0).expect("zdotc not registered")
}

pub(crate) fn get_zdotc_ilp64_ptr() -> *const () {
    if let Some(p) = ZDOTC_ILP64_PTR.get() {
        return p.0;
    }
    ZDOTC_LP64_PTR.get().map(|p| p.0).expect("zdotc not registered")
}

// Internal getters (used by blas2/gemv.rs, blas3/gemm.rs etc.)
// =============================================================================

// BLAS Level 2 getters

#[inline]
pub(crate) fn get_sgemv() -> SgemvFnPtr {
    *SGEMV
        .get()
        .expect("sgemv not registered: call register_sgemv() first")
}

#[inline]
pub(crate) fn get_dgemv() -> DgemvFnPtr {
    *DGEMV
        .get()
        .expect("dgemv not registered: call register_dgemv() first")
}

#[inline]
pub(crate) fn get_cgemv() -> CgemvFnPtr {
    *CGEMV
        .get()
        .expect("cgemv not registered: call register_cgemv() first")
}

#[inline]
pub(crate) fn get_zgemv() -> ZgemvFnPtr {
    *ZGEMV
        .get()
        .expect("zgemv not registered: call register_zgemv() first")
}

#[inline]
pub(crate) fn get_sgbmv() -> SgbmvFnPtr {
    *SGBMV
        .get()
        .expect("sgbmv not registered: call register_sgbmv() first")
}

#[inline]
pub(crate) fn get_dgbmv() -> DgbmvFnPtr {
    *DGBMV
        .get()
        .expect("dgbmv not registered: call register_dgbmv() first")
}

#[inline]
pub(crate) fn get_cgbmv() -> CgbmvFnPtr {
    *CGBMV
        .get()
        .expect("cgbmv not registered: call register_cgbmv() first")
}

#[inline]
pub(crate) fn get_zgbmv() -> ZgbmvFnPtr {
    *ZGBMV
        .get()
        .expect("zgbmv not registered: call register_zgbmv() first")
}

#[inline]
pub(crate) fn get_strmv() -> StrmvFnPtr {
    *STRMV
        .get()
        .expect("strmv not registered: call register_strmv() first")
}

#[inline]
pub(crate) fn get_dtrmv() -> DtrmvFnPtr {
    *DTRMV
        .get()
        .expect("dtrmv not registered: call register_dtrmv() first")
}

#[inline]
pub(crate) fn get_ctrmv() -> CtrmvFnPtr {
    *CTRMV
        .get()
        .expect("ctrmv not registered: call register_ctrmv() first")
}

#[inline]
pub(crate) fn get_ztrmv() -> ZtrmvFnPtr {
    *ZTRMV
        .get()
        .expect("ztrmv not registered: call register_ztrmv() first")
}

#[inline]
pub(crate) fn get_strsv() -> StrsvFnPtr {
    *STRSV
        .get()
        .expect("strsv not registered: call register_strsv() first")
}

#[inline]
pub(crate) fn get_dtrsv() -> DtrsvFnPtr {
    *DTRSV
        .get()
        .expect("dtrsv not registered: call register_dtrsv() first")
}

#[inline]
pub(crate) fn get_ctrsv() -> CtrsvFnPtr {
    *CTRSV
        .get()
        .expect("ctrsv not registered: call register_ctrsv() first")
}

#[inline]
pub(crate) fn get_ztrsv() -> ZtrsvFnPtr {
    *ZTRSV
        .get()
        .expect("ztrsv not registered: call register_ztrsv() first")
}

#[inline]
pub(crate) fn get_stbmv() -> StbmvFnPtr {
    *STBMV
        .get()
        .expect("stbmv not registered: call register_stbmv() first")
}

#[inline]
pub(crate) fn get_dtbmv() -> DtbmvFnPtr {
    *DTBMV
        .get()
        .expect("dtbmv not registered: call register_dtbmv() first")
}

#[inline]
pub(crate) fn get_ctbmv() -> CtbmvFnPtr {
    *CTBMV
        .get()
        .expect("ctbmv not registered: call register_ctbmv() first")
}

#[inline]
pub(crate) fn get_ztbmv() -> ZtbmvFnPtr {
    *ZTBMV
        .get()
        .expect("ztbmv not registered: call register_ztbmv() first")
}

#[inline]
pub(crate) fn get_stbsv() -> StbsvFnPtr {
    *STBSV
        .get()
        .expect("stbsv not registered: call register_stbsv() first")
}

#[inline]
pub(crate) fn get_dtbsv() -> DtbsvFnPtr {
    *DTBSV
        .get()
        .expect("dtbsv not registered: call register_dtbsv() first")
}

#[inline]
pub(crate) fn get_ctbsv() -> CtbsvFnPtr {
    *CTBSV
        .get()
        .expect("ctbsv not registered: call register_ctbsv() first")
}

#[inline]
pub(crate) fn get_ztbsv() -> ZtbsvFnPtr {
    *ZTBSV
        .get()
        .expect("ztbsv not registered: call register_ztbsv() first")
}

#[inline]
pub(crate) fn get_sger() -> SgerFnPtr {
    *SGER
        .get()
        .expect("sger not registered: call register_sger() first")
}

#[inline]
pub(crate) fn get_dger() -> DgerFnPtr {
    *DGER
        .get()
        .expect("dger not registered: call register_dger() first")
}

#[inline]
pub(crate) fn get_cgeru() -> CgeruFnPtr {
    *CGERU
        .get()
        .expect("cgeru not registered: call register_cgeru() first")
}

#[inline]
pub(crate) fn get_cgerc() -> CgercFnPtr {
    *CGERC
        .get()
        .expect("cgerc not registered: call register_cgerc() first")
}

#[inline]
pub(crate) fn get_zgeru() -> ZgeruFnPtr {
    *ZGERU
        .get()
        .expect("zgeru not registered: call register_zgeru() first")
}

#[inline]
pub(crate) fn get_zgerc() -> ZgercFnPtr {
    *ZGERC
        .get()
        .expect("zgerc not registered: call register_zgerc() first")
}

#[inline]
pub(crate) fn get_ssyr() -> SsyrFnPtr {
    *SSYR
        .get()
        .expect("ssyr not registered: call register_ssyr() first")
}

#[inline]
pub(crate) fn get_dsyr() -> DsyrFnPtr {
    *DSYR
        .get()
        .expect("dsyr not registered: call register_dsyr() first")
}

#[inline]
pub(crate) fn get_cher() -> CherFnPtr {
    *CHER
        .get()
        .expect("cher not registered: call register_cher() first")
}

#[inline]
pub(crate) fn get_zher() -> ZherFnPtr {
    *ZHER
        .get()
        .expect("zher not registered: call register_zher() first")
}

#[inline]
pub(crate) fn get_ssyr2() -> Ssyr2FnPtr {
    *SSYR2
        .get()
        .expect("ssyr2 not registered: call register_ssyr2() first")
}

#[inline]
pub(crate) fn get_dsyr2() -> Dsyr2FnPtr {
    *DSYR2
        .get()
        .expect("dsyr2 not registered: call register_dsyr2() first")
}

#[inline]
pub(crate) fn get_cher2() -> Cher2FnPtr {
    *CHER2
        .get()
        .expect("cher2 not registered: call register_cher2() first")
}

#[inline]
pub(crate) fn get_zher2() -> Zher2FnPtr {
    *ZHER2
        .get()
        .expect("zher2 not registered: call register_zher2() first")
}

// BLAS Level 2 packed matrix getters

#[inline]
pub(crate) fn get_sspmv() -> SspmvFnPtr {
    *SSPMV
        .get()
        .expect("sspmv not registered: call register_sspmv() first")
}

#[inline]
pub(crate) fn get_dspmv() -> DspmvFnPtr {
    *DSPMV
        .get()
        .expect("dspmv not registered: call register_dspmv() first")
}

#[inline]
pub(crate) fn get_chpmv() -> ChpmvFnPtr {
    *CHPMV
        .get()
        .expect("chpmv not registered: call register_chpmv() first")
}

#[inline]
pub(crate) fn get_zhpmv() -> ZhpmvFnPtr {
    *ZHPMV
        .get()
        .expect("zhpmv not registered: call register_zhpmv() first")
}

#[inline]
pub(crate) fn get_stpmv() -> StpmvFnPtr {
    *STPMV
        .get()
        .expect("stpmv not registered: call register_stpmv() first")
}

#[inline]
pub(crate) fn get_dtpmv() -> DtpmvFnPtr {
    *DTPMV
        .get()
        .expect("dtpmv not registered: call register_dtpmv() first")
}

#[inline]
pub(crate) fn get_ctpmv() -> CtpmvFnPtr {
    *CTPMV
        .get()
        .expect("ctpmv not registered: call register_ctpmv() first")
}

#[inline]
pub(crate) fn get_ztpmv() -> ZtpmvFnPtr {
    *ZTPMV
        .get()
        .expect("ztpmv not registered: call register_ztpmv() first")
}

#[inline]
pub(crate) fn get_stpsv() -> StpsvFnPtr {
    *STPSV
        .get()
        .expect("stpsv not registered: call register_stpsv() first")
}

#[inline]
pub(crate) fn get_dtpsv() -> DtpsvFnPtr {
    *DTPSV
        .get()
        .expect("dtpsv not registered: call register_dtpsv() first")
}

#[inline]
pub(crate) fn get_ctpsv() -> CtpsvFnPtr {
    *CTPSV
        .get()
        .expect("ctpsv not registered: call register_ctpsv() first")
}

#[inline]
pub(crate) fn get_ztpsv() -> ZtpsvFnPtr {
    *ZTPSV
        .get()
        .expect("ztpsv not registered: call register_ztpsv() first")
}

#[inline]
pub(crate) fn get_sspr() -> SsprFnPtr {
    *SSPR
        .get()
        .expect("sspr not registered: call register_sspr() first")
}

#[inline]
pub(crate) fn get_dspr() -> DsprFnPtr {
    *DSPR
        .get()
        .expect("dspr not registered: call register_dspr() first")
}

#[inline]
pub(crate) fn get_chpr() -> ChprFnPtr {
    *CHPR
        .get()
        .expect("chpr not registered: call register_chpr() first")
}

#[inline]
pub(crate) fn get_zhpr() -> ZhprFnPtr {
    *ZHPR
        .get()
        .expect("zhpr not registered: call register_zhpr() first")
}

#[inline]
pub(crate) fn get_sspr2() -> Sspr2FnPtr {
    *SSPR2
        .get()
        .expect("sspr2 not registered: call register_sspr2() first")
}

#[inline]
pub(crate) fn get_dspr2() -> Dspr2FnPtr {
    *DSPR2
        .get()
        .expect("dspr2 not registered: call register_dspr2() first")
}

#[inline]
pub(crate) fn get_chpr2() -> Chpr2FnPtr {
    *CHPR2
        .get()
        .expect("chpr2 not registered: call register_chpr2() first")
}

#[inline]
pub(crate) fn get_zhpr2() -> Zhpr2FnPtr {
    *ZHPR2
        .get()
        .expect("zhpr2 not registered: call register_zhpr2() first")
}

// BLAS Level 3 getters

#[inline]
#[allow(dead_code)]
pub(crate) fn get_dgemm() -> DgemmFnPtr {
    *DGEMM
        .get()
        .expect("dgemm not registered: call register_dgemm() first")
}

#[cfg(not(feature = "ilp64"))]
#[inline]
pub(crate) fn get_dgemm_for_current_cblas() -> DgemmProvider {
    if let Some(f) = DGEMM_LP64.get() {
        DgemmProvider::Lp64(*f)
    } else if let Some(f) = DGEMM_ILP64.get() {
        DgemmProvider::Ilp64(*f)
    } else {
        panic!(
            "dgemm not registered for current CBLAS ABI: call register_dgemm(), \
             cblas_inject_register_dgemm_lp64(), or cblas_inject_register_dgemm_ilp64() first"
        );
    }
}

#[cfg(feature = "ilp64")]
#[inline]
pub(crate) fn get_dgemm_for_current_cblas() -> DgemmProvider {
    if let Some(f) = DGEMM_ILP64.get() {
        DgemmProvider::Ilp64(*f)
    } else if let Some(f) = DGEMM_LP64.get() {
        DgemmProvider::Lp64(*f)
    } else {
        panic!(
            "dgemm not registered for current CBLAS ABI: call register_dgemm(), \
             cblas_inject_register_dgemm_ilp64(), or cblas_inject_register_dgemm_lp64() first"
        );
    }
}

#[inline]
pub(crate) fn get_dgemm_for_ilp64_cblas() -> DgemmProvider {
    if let Some(f) = DGEMM_ILP64.get() {
        DgemmProvider::Ilp64(*f)
    } else if let Some(f) = DGEMM_LP64.get() {
        DgemmProvider::Lp64(*f)
    } else {
        panic!(
            "dgemm not registered for ILP64 CBLAS ABI: call register_dgemm(), \
             cblas_inject_register_dgemm_ilp64(), or cblas_inject_register_dgemm_lp64() first"
        );
    }
}

#[inline]
pub(crate) fn get_sgemm() -> SgemmFnPtr {
    *SGEMM
        .get()
        .expect("sgemm not registered: call register_sgemm() first")
}

#[cfg(not(feature = "ilp64"))]
#[inline]
pub(crate) fn get_sgemm_for_current_cblas() -> SgemmProvider {
    if let Some(f) = SGEMM_LP64.get() {
        SgemmProvider::Lp64(*f)
    } else if let Some(f) = SGEMM_ILP64.get() {
        SgemmProvider::Ilp64(*f)
    } else {
        panic!(
            "sgemm not registered: call register_sgemm() or cblas_inject_register_sgemm_lp64()"
        );
    }
}

#[cfg(feature = "ilp64")]
#[inline]
pub(crate) fn get_sgemm_for_current_cblas() -> SgemmProvider {
    if let Some(f) = SGEMM_ILP64.get() {
        SgemmProvider::Ilp64(*f)
    } else if let Some(f) = SGEMM_LP64.get() {
        SgemmProvider::Lp64(*f)
    } else {
        panic!(
            "sgemm not registered: call register_sgemm() or cblas_inject_register_sgemm_ilp64()"
        );
    }
}

#[inline]
pub(crate) fn get_sgemm_for_ilp64_cblas() -> SgemmProvider {
    if let Some(f) = SGEMM_ILP64.get() {
        SgemmProvider::Ilp64(*f)
    } else if let Some(f) = SGEMM_LP64.get() {
        SgemmProvider::Lp64(*f)
    } else {
        panic!("sgemm not registered for ILP64 CBLAS ABI");
    }
}

#[inline]
#[allow(dead_code)]
pub(crate) fn get_zgemm() -> ZgemmFnPtr {
    *ZGEMM
        .get()
        .expect("zgemm not registered: call register_zgemm() first")
}

#[cfg(not(feature = "ilp64"))]
#[inline]
pub(crate) fn get_zgemm_for_current_cblas() -> ZgemmProvider {
    if let Some(f) = ZGEMM_LP64.get() {
        ZgemmProvider::Lp64(*f)
    } else if let Some(f) = ZGEMM_ILP64.get() {
        ZgemmProvider::Ilp64(*f)
    } else {
        panic!(
            "zgemm not registered for current CBLAS ABI: call register_zgemm(), \
             cblas_inject_register_zgemm_lp64(), or cblas_inject_register_zgemm_ilp64() first"
        );
    }
}

#[cfg(feature = "ilp64")]
#[inline]
pub(crate) fn get_zgemm_for_current_cblas() -> ZgemmProvider {
    if let Some(f) = ZGEMM_ILP64.get() {
        ZgemmProvider::Ilp64(*f)
    } else if let Some(f) = ZGEMM_LP64.get() {
        ZgemmProvider::Lp64(*f)
    } else {
        panic!(
            "zgemm not registered for current CBLAS ABI: call register_zgemm(), \
             cblas_inject_register_zgemm_ilp64(), or cblas_inject_register_zgemm_lp64() first"
        );
    }
}

#[inline]
pub(crate) fn get_zgemm_for_ilp64_cblas() -> ZgemmProvider {
    if let Some(f) = ZGEMM_ILP64.get() {
        ZgemmProvider::Ilp64(*f)
    } else if let Some(f) = ZGEMM_LP64.get() {
        ZgemmProvider::Lp64(*f)
    } else {
        panic!(
            "zgemm not registered for ILP64 CBLAS ABI: call register_zgemm(), \
             cblas_inject_register_zgemm_ilp64(), or cblas_inject_register_zgemm_lp64() first"
        );
    }
}

#[inline]
pub(crate) fn get_cgemm() -> CgemmFnPtr {
    *CGEMM
        .get()
        .expect("cgemm not registered: call register_cgemm() first")
}

#[cfg(not(feature = "ilp64"))]
#[inline]
pub(crate) fn get_cgemm_for_current_cblas() -> CgemmProvider {
    if let Some(f) = CGEMM_LP64.get() {
        CgemmProvider::Lp64(*f)
    } else if let Some(f) = CGEMM_ILP64.get() {
        CgemmProvider::Ilp64(*f)
    } else {
        panic!(
            "cgemm not registered: call register_cgemm() or cblas_inject_register_cgemm_lp64()"
        );
    }
}

#[cfg(feature = "ilp64")]
#[inline]
pub(crate) fn get_cgemm_for_current_cblas() -> CgemmProvider {
    if let Some(f) = CGEMM_ILP64.get() {
        CgemmProvider::Ilp64(*f)
    } else if let Some(f) = CGEMM_LP64.get() {
        CgemmProvider::Lp64(*f)
    } else {
        panic!(
            "cgemm not registered: call register_cgemm() or cblas_inject_register_cgemm_ilp64()"
        );
    }
}

#[inline]
pub(crate) fn get_cgemm_for_ilp64_cblas() -> CgemmProvider {
    if let Some(f) = CGEMM_ILP64.get() {
        CgemmProvider::Ilp64(*f)
    } else if let Some(f) = CGEMM_LP64.get() {
        CgemmProvider::Lp64(*f)
    } else {
        panic!("cgemm not registered for ILP64 CBLAS ABI");
    }
}

#[inline]
pub(crate) fn get_ssymm() -> SsymmFnPtr {
    *SSYMM
        .get()
        .expect("ssymm not registered: call register_ssymm() first")
}

#[inline]
pub(crate) fn get_dsymm() -> DsymmFnPtr {
    *DSYMM
        .get()
        .expect("dsymm not registered: call register_dsymm() first")
}

#[inline]
pub(crate) fn get_csymm() -> CsymmFnPtr {
    *CSYMM
        .get()
        .expect("csymm not registered: call register_csymm() first")
}

#[inline]
pub(crate) fn get_zsymm() -> ZsymmFnPtr {
    *ZSYMM
        .get()
        .expect("zsymm not registered: call register_zsymm() first")
}

#[inline]
pub(crate) fn get_chemm() -> ChemmFnPtr {
    *CHEMM
        .get()
        .expect("chemm not registered: call register_chemm() first")
}

#[inline]
pub(crate) fn get_zhemm() -> ZhemmFnPtr {
    *ZHEMM
        .get()
        .expect("zhemm not registered: call register_zhemm() first")
}

#[inline]
pub(crate) fn get_dsyrk() -> DsyrkFnPtr {
    *DSYRK
        .get()
        .expect("dsyrk not registered: call register_dsyrk() first")
}

#[inline]
pub(crate) fn get_ssyrk() -> SsyrkFnPtr {
    *SSYRK
        .get()
        .expect("ssyrk not registered: call register_ssyrk() first")
}

#[inline]
pub(crate) fn get_csyrk() -> CsyrkFnPtr {
    *CSYRK
        .get()
        .expect("csyrk not registered: call register_csyrk() first")
}

#[inline]
pub(crate) fn get_zsyrk() -> ZsyrkFnPtr {
    *ZSYRK
        .get()
        .expect("zsyrk not registered: call register_zsyrk() first")
}

#[inline]
pub(crate) fn get_cherk() -> CherkFnPtr {
    *CHERK
        .get()
        .expect("cherk not registered: call register_cherk() first")
}

#[inline]
pub(crate) fn get_zherk() -> ZherkFnPtr {
    *ZHERK
        .get()
        .expect("zherk not registered: call register_zherk() first")
}

#[inline]
pub(crate) fn get_dsyr2k() -> Dsyr2kFnPtr {
    *DSYR2K
        .get()
        .expect("dsyr2k not registered: call register_dsyr2k() first")
}

#[inline]
pub(crate) fn get_ssyr2k() -> Ssyr2kFnPtr {
    *SSYR2K
        .get()
        .expect("ssyr2k not registered: call register_ssyr2k() first")
}

#[inline]
pub(crate) fn get_csyr2k() -> Csyr2kFnPtr {
    *CSYR2K
        .get()
        .expect("csyr2k not registered: call register_csyr2k() first")
}

#[inline]
pub(crate) fn get_zsyr2k() -> Zsyr2kFnPtr {
    *ZSYR2K
        .get()
        .expect("zsyr2k not registered: call register_zsyr2k() first")
}

#[inline]
pub(crate) fn get_cher2k() -> Cher2kFnPtr {
    *CHER2K
        .get()
        .expect("cher2k not registered: call register_cher2k() first")
}

#[inline]
pub(crate) fn get_zher2k() -> Zher2kFnPtr {
    *ZHER2K
        .get()
        .expect("zher2k not registered: call register_zher2k() first")
}

#[inline]
pub(crate) fn get_dtrmm() -> DtrmmFnPtr {
    *DTRMM
        .get()
        .expect("dtrmm not registered: call register_dtrmm() first")
}

#[inline]
pub(crate) fn get_dtrsm() -> DtrsmFnPtr {
    *DTRSM
        .get()
        .expect("dtrsm not registered: call register_dtrsm() first")
}


#[inline]
pub(crate) fn get_strmm() -> StrmmFnPtr {
    *STRMM
        .get()
        .expect("strmm not registered: call register_strmm() first")
}

#[inline]
pub(crate) fn get_ctrmm() -> CtrmmFnPtr {
    *CTRMM
        .get()
        .expect("ctrmm not registered: call register_ctrmm() first")
}

#[inline]
pub(crate) fn get_ztrmm() -> ZtrmmFnPtr {
    *ZTRMM
        .get()
        .expect("ztrmm not registered: call register_ztrmm() first")
}

#[inline]
pub(crate) fn get_strsm() -> StrsmFnPtr {
    *STRSM
        .get()
        .expect("strsm not registered: call register_strsm() first")
}

#[inline]
pub(crate) fn get_ctrsm() -> CtrsmFnPtr {
    *CTRSM
        .get()
        .expect("ctrsm not registered: call register_ctrsm() first")
}

#[inline]
pub(crate) fn get_ztrsm() -> ZtrsmFnPtr {
    *ZTRSM
        .get()
        .expect("ztrsm not registered: call register_ztrsm() first")
}
// BLAS Level 2 getters

#[inline]
pub(crate) fn get_ssymv() -> SsymvFnPtr {
    *SSYMV
        .get()
        .expect("ssymv not registered: call register_ssymv() first")
}

#[inline]
pub(crate) fn get_dsymv() -> DsymvFnPtr {
    *DSYMV
        .get()
        .expect("dsymv not registered: call register_dsymv() first")
}

#[inline]
pub(crate) fn get_chemv() -> ChemvFnPtr {
    *CHEMV
        .get()
        .expect("chemv not registered: call register_chemv() first")
}

#[inline]
pub(crate) fn get_zhemv() -> ZhemvFnPtr {
    *ZHEMV
        .get()
        .expect("zhemv not registered: call register_zhemv() first")
}

#[inline]
pub(crate) fn get_ssbmv() -> SsbmvFnPtr {
    *SSBMV
        .get()
        .expect("ssbmv not registered: call register_ssbmv() first")
}

#[inline]
pub(crate) fn get_dsbmv() -> DsbmvFnPtr {
    *DSBMV
        .get()
        .expect("dsbmv not registered: call register_dsbmv() first")
}

#[inline]
pub(crate) fn get_chbmv() -> ChbmvFnPtr {
    *CHBMV
        .get()
        .expect("chbmv not registered: call register_chbmv() first")
}

#[inline]
pub(crate) fn get_zhbmv() -> ZhbmvFnPtr {
    *ZHBMV
        .get()
        .expect("zhbmv not registered: call register_zhbmv() first")
}

#[inline]
pub(crate) fn get_srot() -> SrotFnPtr {
    *SROT
        .get()
        .expect("srot not registered: call register_srot() first")
}

#[inline]
pub(crate) fn get_drot() -> DrotFnPtr {
    *DROT
        .get()
        .expect("drot not registered: call register_drot() first")
}

#[inline]
pub(crate) fn get_srotg() -> SrotgFnPtr {
    *SROTG
        .get()
        .expect("srotg not registered: call register_srotg() first")
}

#[inline]
pub(crate) fn get_drotg() -> DrotgFnPtr {
    *DROTG
        .get()
        .expect("drotg not registered: call register_drotg() first")
}

#[inline]
pub(crate) fn get_srotm() -> SrotmFnPtr {
    *SROTM
        .get()
        .expect("srotm not registered: call register_srotm() first")
}

#[inline]
pub(crate) fn get_drotm() -> DrotmFnPtr {
    *DROTM
        .get()
        .expect("drotm not registered: call register_drotm() first")
}

#[inline]
pub(crate) fn get_srotmg() -> SrotmgFnPtr {
    *SROTMG
        .get()
        .expect("srotmg not registered: call register_srotmg() first")
}

#[inline]
pub(crate) fn get_drotmg() -> DrotmgFnPtr {
    *DROTMG
        .get()
        .expect("drotmg not registered: call register_drotmg() first")
}

#[inline]
pub(crate) fn get_scabs1() -> Scabs1FnPtr {
    *SCABS1
        .get()
        .expect("scabs1 not registered: call register_scabs1() first")
}

#[inline]
pub(crate) fn get_dcabs1() -> Dcabs1FnPtr {
    *DCABS1
        .get()
        .expect("dcabs1 not registered: call register_dcabs1() first")
}

#[inline]
pub(crate) fn get_sswap() -> SswapFnPtr {
    *SSWAP
        .get()
        .expect("sswap not registered: call register_sswap() first")
}

#[inline]
pub(crate) fn get_dswap() -> DswapFnPtr {
    *DSWAP
        .get()
        .expect("dswap not registered: call register_dswap() first")
}

#[inline]
pub(crate) fn get_cswap() -> CswapFnPtr {
    *CSWAP
        .get()
        .expect("cswap not registered: call register_cswap() first")
}

#[inline]
pub(crate) fn get_zswap() -> ZswapFnPtr {
    *ZSWAP
        .get()
        .expect("zswap not registered: call register_zswap() first")
}

#[inline]
pub(crate) fn get_scopy() -> ScopyFnPtr {
    *SCOPY
        .get()
        .expect("scopy not registered: call register_scopy() first")
}

#[inline]
pub(crate) fn get_dcopy() -> DcopyFnPtr {
    *DCOPY
        .get()
        .expect("dcopy not registered: call register_dcopy() first")
}

#[inline]
pub(crate) fn get_ccopy() -> CcopyFnPtr {
    *CCOPY
        .get()
        .expect("ccopy not registered: call register_ccopy() first")
}

#[inline]
pub(crate) fn get_zcopy() -> ZcopyFnPtr {
    *ZCOPY
        .get()
        .expect("zcopy not registered: call register_zcopy() first")
}

#[inline]
pub(crate) fn get_saxpy() -> SaxpyFnPtr {
    *SAXPY
        .get()
        .expect("saxpy not registered: call register_saxpy() first")
}

#[inline]
pub(crate) fn get_daxpy() -> DaxpyFnPtr {
    *DAXPY
        .get()
        .expect("daxpy not registered: call register_daxpy() first")
}

#[inline]
pub(crate) fn get_caxpy() -> CaxpyFnPtr {
    *CAXPY
        .get()
        .expect("caxpy not registered: call register_caxpy() first")
}

#[inline]
pub(crate) fn get_zaxpy() -> ZaxpyFnPtr {
    *ZAXPY
        .get()
        .expect("zaxpy not registered: call register_zaxpy() first")
}

#[inline]
pub(crate) fn get_sscal() -> SscalFnPtr {
    *SSCAL
        .get()
        .expect("sscal not registered: call register_sscal() first")
}

#[inline]
pub(crate) fn get_dscal() -> DscalFnPtr {
    *DSCAL
        .get()
        .expect("dscal not registered: call register_dscal() first")
}

#[inline]
pub(crate) fn get_cscal() -> CscalFnPtr {
    *CSCAL
        .get()
        .expect("cscal not registered: call register_cscal() first")
}

#[inline]
pub(crate) fn get_zscal() -> ZscalFnPtr {
    *ZSCAL
        .get()
        .expect("zscal not registered: call register_zscal() first")
}

#[inline]
pub(crate) fn get_csscal() -> CsscalFnPtr {
    *CSSCAL
        .get()
        .expect("csscal not registered: call register_csscal() first")
}

#[inline]
pub(crate) fn get_zdscal() -> ZdscalFnPtr {
    *ZDSCAL
        .get()
        .expect("zdscal not registered: call register_zdscal() first")
}

#[inline]
pub(crate) fn get_sdot() -> SdotFnPtr {
    *SDOT
        .get()
        .expect("sdot not registered: call register_sdot() first")
}

#[inline]
pub(crate) fn get_ddot() -> DdotFnPtr {
    *DDOT
        .get()
        .expect("ddot not registered: call register_ddot() first")
}

#[inline]
pub(crate) fn get_cdotu_ptr() -> *const () {
    get_cdotu_lp64_ptr()
}

#[inline]
pub(crate) fn get_zdotu_ptr() -> *const () {
    get_zdotu_lp64_ptr()
}

#[inline]
pub(crate) fn get_cdotc_ptr() -> *const () {
    get_cdotc_lp64_ptr()
}

#[inline]
pub(crate) fn get_zdotc_ptr() -> *const () {
    get_zdotc_lp64_ptr()
}

#[inline]
pub(crate) fn get_sdsdot() -> SdsdotFnPtr {
    *SDSDOT
        .get()
        .expect("sdsdot not registered: call register_sdsdot() first")
}

#[inline]
pub(crate) fn get_dsdot() -> DsdotFnPtr {
    *DSDOT
        .get()
        .expect("dsdot not registered: call register_dsdot() first")
}

#[inline]
pub(crate) fn get_snrm2() -> Snrm2FnPtr {
    *SNRM2
        .get()
        .expect("snrm2 not registered: call register_snrm2() first")
}

#[inline]
pub(crate) fn get_dnrm2() -> Dnrm2FnPtr {
    *DNRM2
        .get()
        .expect("dnrm2 not registered: call register_dnrm2() first")
}

#[inline]
pub(crate) fn get_scnrm2() -> Scnrm2FnPtr {
    *SCNRM2
        .get()
        .expect("scnrm2 not registered: call register_scnrm2() first")
}

#[inline]
pub(crate) fn get_dznrm2() -> Dznrm2FnPtr {
    *DZNRM2
        .get()
        .expect("dznrm2 not registered: call register_dznrm2() first")
}

#[inline]
pub(crate) fn get_sasum() -> SasumFnPtr {
    *SASUM
        .get()
        .expect("sasum not registered: call register_sasum() first")
}

#[inline]
pub(crate) fn get_dasum() -> DasumFnPtr {
    *DASUM
        .get()
        .expect("dasum not registered: call register_dasum() first")
}

#[inline]
pub(crate) fn get_scasum() -> ScasumFnPtr {
    *SCASUM
        .get()
        .expect("scasum not registered: call register_scasum() first")
}

#[inline]
pub(crate) fn get_dzasum() -> DzasumFnPtr {
    *DZASUM
        .get()
        .expect("dzasum not registered: call register_dzasum() first")
}

#[inline]
pub(crate) fn get_isamax() -> IsamaxFnPtr {
    *ISAMAX
        .get()
        .expect("isamax not registered: call register_isamax() first")
}

#[inline]
pub(crate) fn get_idamax() -> IdamaxFnPtr {
    *IDAMAX
        .get()
        .expect("idamax not registered: call register_idamax() first")
}

#[inline]
pub(crate) fn get_icamax() -> IcamaxFnPtr {
    *ICAMAX
        .get()
        .expect("icamax not registered: call register_icamax() first")
}

#[inline]
pub(crate) fn get_izamax() -> IzamaxFnPtr {
    *IZAMAX
        .get()
        .expect("izamax not registered: call register_izamax() first")
}

// =============================================================================
// Query functions
// =============================================================================

/// Check if dgemm is registered.
pub fn is_dgemm_registered() -> bool {
    DGEMM.get().is_some()
}

/// Check if sgemm is registered.
pub fn is_sgemm_registered() -> bool {
    SGEMM.get().is_some()
}

/// Check if zgemm is registered.
pub fn is_zgemm_registered() -> bool {
    ZGEMM.get().is_some()
}

/// Check if cgemm is registered.
pub fn is_cgemm_registered() -> bool {
    CGEMM.get().is_some()
}
