//! Fortran BLAS/LAPACK function pointer registration.
//!
//! This module provides the infrastructure for registering Fortran BLAS/LAPACK
//! function pointers at runtime. Each function has its own `OnceLock` to allow
//! partial registration (only register the functions you need).

use std::ffi::c_char;
use std::sync::OnceLock;

use num_complex::{Complex32, Complex64};

use crate::blasint;

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

/// Fortran dswap function pointer type (double precision vector swap)
pub type DswapFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut f64,
    incx: *const blasint,
    y: *mut f64,
    incy: *const blasint,
);

/// Fortran cswap function pointer type (single precision complex vector swap)
pub type CswapFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut Complex32,
    incx: *const blasint,
    y: *mut Complex32,
    incy: *const blasint,
);

/// Fortran zswap function pointer type (double precision complex vector swap)
pub type ZswapFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut Complex64,
    incx: *const blasint,
    y: *mut Complex64,
    incy: *const blasint,
);

/// Fortran scopy function pointer type (single precision vector copy)
pub type ScopyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const f32,
    incx: *const blasint,
    y: *mut f32,
    incy: *const blasint,
);

/// Fortran dcopy function pointer type (double precision vector copy)
pub type DcopyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const f64,
    incx: *const blasint,
    y: *mut f64,
    incy: *const blasint,
);

/// Fortran ccopy function pointer type (single precision complex vector copy)
pub type CcopyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    y: *mut Complex32,
    incy: *const blasint,
);

/// Fortran zcopy function pointer type (double precision complex vector copy)
pub type ZcopyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    y: *mut Complex64,
    incy: *const blasint,
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

/// Fortran daxpy function pointer type (double precision y = alpha*x + y)
pub type DaxpyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const f64,
    x: *const f64,
    incx: *const blasint,
    y: *mut f64,
    incy: *const blasint,
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

/// Fortran zaxpy function pointer type (double precision complex y = alpha*x + y)
pub type ZaxpyFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: *const blasint,
    y: *mut Complex64,
    incy: *const blasint,
);

/// Fortran sscal function pointer type (single precision vector scaling)
pub type SscalFnPtr =
    unsafe extern "C" fn(n: *const blasint, alpha: *const f32, x: *mut f32, incx: *const blasint);

/// Fortran dscal function pointer type (double precision vector scaling)
pub type DscalFnPtr =
    unsafe extern "C" fn(n: *const blasint, alpha: *const f64, x: *mut f64, incx: *const blasint);

/// Fortran cscal function pointer type (single precision complex vector scaling)
pub type CscalFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const Complex32,
    x: *mut Complex32,
    incx: *const blasint,
);

/// Fortran zscal function pointer type (double precision complex vector scaling)
pub type ZscalFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const Complex64,
    x: *mut Complex64,
    incx: *const blasint,
);

/// Fortran csscal function pointer type (scale complex vector by real scalar)
pub type CsscalFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const f32,
    x: *mut Complex32,
    incx: *const blasint,
);

/// Fortran zdscal function pointer type (scale complex vector by real scalar)
pub type ZdscalFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    alpha: *const f64,
    x: *mut Complex64,
    incx: *const blasint,
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

/// Fortran drotg function pointer type (generate Givens rotation, double precision)
pub type DrotgFnPtr = unsafe extern "C" fn(a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64);

/// Fortran srotg function pointer type (generate Givens rotation, single precision)
pub type SrotgFnPtr = unsafe extern "C" fn(a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32);

/// Fortran drotm function pointer type (apply modified Givens rotation, double precision)
pub type DrotmFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *mut f64,
    incx: *const blasint,
    y: *mut f64,
    incy: *const blasint,
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

/// Fortran drotmg function pointer type (generate modified Givens rotation, double precision)
pub type DrotmgFnPtr =
    unsafe extern "C" fn(d1: *mut f64, d2: *mut f64, b1: *mut f64, b2: *const f64, p: *mut f64);

/// Fortran srotmg function pointer type (generate modified Givens rotation, single precision)
pub type SrotmgFnPtr =
    unsafe extern "C" fn(d1: *mut f32, d2: *mut f32, b1: *mut f32, b2: *const f32, p: *mut f32);

/// Fortran dcabs1 function pointer type (|Re(z)| + |Im(z)|, double precision complex)
pub type Dcabs1FnPtr = unsafe extern "C" fn(z: *const Complex64) -> f64;

/// Fortran scabs1 function pointer type (|Re(z)| + |Im(z)|, single precision complex)
pub type Scabs1FnPtr = unsafe extern "C" fn(z: *const Complex32) -> f32;

/// Fortran sdot function pointer type (single precision dot product)
pub type SdotFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const f32,
    incx: *const blasint,
    y: *const f32,
    incy: *const blasint,
) -> f32;

/// Fortran ddot function pointer type (double precision dot product)
pub type DdotFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const f64,
    incx: *const blasint,
    y: *const f64,
    incy: *const blasint,
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

/// Fortran zdotu function pointer type (complex double precision dot product, unconjugated)
/// Return value convention: complex returned via register (OpenBLAS, MKL intel, BLIS)
pub type ZdotuFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    y: *const Complex64,
    incy: *const blasint,
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

/// Fortran cdotc function pointer type (complex single precision dot product, conjugated)
/// Return value convention: complex returned via register (OpenBLAS, MKL intel, BLIS)
pub type CdotcFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex32,
    incx: *const blasint,
    y: *const Complex32,
    incy: *const blasint,
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

/// Fortran zdotc function pointer type (complex double precision dot product, conjugated)
/// Return value convention: complex returned via register (OpenBLAS, MKL intel, BLIS)
pub type ZdotcFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const Complex64,
    incx: *const blasint,
    y: *const Complex64,
    incy: *const blasint,
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

/// Fortran sdsdot function pointer type (single precision dot product with double precision accumulation)
pub type SdsdotFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    sb: *const f32,
    x: *const f32,
    incx: *const blasint,
    y: *const f32,
    incy: *const blasint,
) -> f32;

/// Fortran dsdot function pointer type (double precision dot product of single precision vectors)
pub type DsdotFnPtr = unsafe extern "C" fn(
    n: *const blasint,
    x: *const f32,
    incx: *const blasint,
    y: *const f32,
    incy: *const blasint,
) -> f64;

/// Fortran snrm2 function pointer type (single precision Euclidean norm)
pub type Snrm2FnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f32, incx: *const blasint) -> f32;

/// Fortran dnrm2 function pointer type (double precision Euclidean norm)
pub type Dnrm2FnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f64, incx: *const blasint) -> f64;

/// Fortran scnrm2 function pointer type (complex single precision Euclidean norm)
pub type Scnrm2FnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex32, incx: *const blasint) -> f32;

/// Fortran dznrm2 function pointer type (complex double precision Euclidean norm)
pub type Dznrm2FnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex64, incx: *const blasint) -> f64;

/// Fortran sasum function pointer type (single precision sum of absolute values)
pub type SasumFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f32, incx: *const blasint) -> f32;

/// Fortran dasum function pointer type (double precision sum of absolute values)
pub type DasumFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f64, incx: *const blasint) -> f64;

/// Fortran scasum function pointer type (complex single precision sum of absolute values)
pub type ScasumFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex32, incx: *const blasint) -> f32;

/// Fortran dzasum function pointer type (complex double precision sum of absolute values)
pub type DzasumFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex64, incx: *const blasint) -> f64;

/// Fortran isamax function pointer type (index of max absolute value, single precision)
pub type IsamaxFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f32, incx: *const blasint) -> blasint;

/// Fortran idamax function pointer type (index of max absolute value, double precision)
pub type IdamaxFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const f64, incx: *const blasint) -> blasint;

/// Fortran icamax function pointer type (index of max absolute value, complex single precision)
pub type IcamaxFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex32, incx: *const blasint) -> blasint;

/// Fortran izamax function pointer type (index of max absolute value, complex double precision)
pub type IzamaxFnPtr =
    unsafe extern "C" fn(n: *const blasint, x: *const Complex64, incx: *const blasint) -> blasint;

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
/// # Panics
///
/// Panics if the style has already been set.
pub fn set_complex_return_style(style: ComplexReturnStyle) {
    COMPLEX_RETURN_STYLE
        .set(style)
        .expect("complex return style already set (can only be set once)");
}

/// Get the current complex return style.
///
/// Returns `ReturnValue` as the default if not explicitly set.
#[inline]
pub fn get_complex_return_style() -> ComplexReturnStyle {
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

static CDOTU_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static ZDOTU_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static CDOTC_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
static ZDOTC_PTR: OnceLock<FnPtrWrapper> = OnceLock::new();
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

// BLAS Level 3
static DGEMM: OnceLock<DgemmFnPtr> = OnceLock::new();
static SGEMM: OnceLock<SgemmFnPtr> = OnceLock::new();
static ZGEMM: OnceLock<ZgemmFnPtr> = OnceLock::new();
static CGEMM: OnceLock<CgemmFnPtr> = OnceLock::new();
static DSYMM: OnceLock<DsymmFnPtr> = OnceLock::new();
static DSYRK: OnceLock<DsyrkFnPtr> = OnceLock::new();
static DSYR2K: OnceLock<Dsyr2kFnPtr> = OnceLock::new();
static DTRMM: OnceLock<DtrmmFnPtr> = OnceLock::new();
static DTRSM: OnceLock<DtrsmFnPtr> = OnceLock::new();

// =============================================================================
// Registration functions
// =============================================================================

// BLAS Level 1 registration

/// Register the Fortran sswap function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran sswap implementation.
pub unsafe fn register_sswap(f: SswapFnPtr) {
    SSWAP
        .set(f)
        .expect("sswap already registered (can only be set once)");
}

/// Register the Fortran dswap function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dswap implementation.
pub unsafe fn register_dswap(f: DswapFnPtr) {
    DSWAP
        .set(f)
        .expect("dswap already registered (can only be set once)");
}

/// Register the Fortran cswap function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran cswap implementation.
pub unsafe fn register_cswap(f: CswapFnPtr) {
    CSWAP
        .set(f)
        .expect("cswap already registered (can only be set once)");
}

/// Register the Fortran zswap function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran zswap implementation.
pub unsafe fn register_zswap(f: ZswapFnPtr) {
    ZSWAP
        .set(f)
        .expect("zswap already registered (can only be set once)");
}

/// Register the Fortran scopy function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran scopy implementation.
pub unsafe fn register_scopy(f: ScopyFnPtr) {
    SCOPY
        .set(f)
        .expect("scopy already registered (can only be set once)");
}

/// Register the Fortran dcopy function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dcopy implementation.
pub unsafe fn register_dcopy(f: DcopyFnPtr) {
    DCOPY
        .set(f)
        .expect("dcopy already registered (can only be set once)");
}

/// Register the Fortran ccopy function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran ccopy implementation.
pub unsafe fn register_ccopy(f: CcopyFnPtr) {
    CCOPY
        .set(f)
        .expect("ccopy already registered (can only be set once)");
}

/// Register the Fortran zcopy function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran zcopy implementation.
pub unsafe fn register_zcopy(f: ZcopyFnPtr) {
    ZCOPY
        .set(f)
        .expect("zcopy already registered (can only be set once)");
}

/// Register the Fortran saxpy function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran saxpy implementation.
pub unsafe fn register_saxpy(f: SaxpyFnPtr) {
    SAXPY
        .set(f)
        .expect("saxpy already registered (can only be set once)");
}

/// Register the Fortran daxpy function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran daxpy implementation.
pub unsafe fn register_daxpy(f: DaxpyFnPtr) {
    DAXPY
        .set(f)
        .expect("daxpy already registered (can only be set once)");
}

/// Register the Fortran caxpy function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran caxpy implementation.
pub unsafe fn register_caxpy(f: CaxpyFnPtr) {
    CAXPY
        .set(f)
        .expect("caxpy already registered (can only be set once)");
}

/// Register the Fortran zaxpy function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran zaxpy implementation.
pub unsafe fn register_zaxpy(f: ZaxpyFnPtr) {
    ZAXPY
        .set(f)
        .expect("zaxpy already registered (can only be set once)");
}

/// Register the Fortran sscal function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran sscal implementation.
pub unsafe fn register_sscal(f: SscalFnPtr) {
    SSCAL
        .set(f)
        .expect("sscal already registered (can only be set once)");
}

/// Register the Fortran dscal function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dscal implementation.
pub unsafe fn register_dscal(f: DscalFnPtr) {
    DSCAL
        .set(f)
        .expect("dscal already registered (can only be set once)");
}

/// Register the Fortran cscal function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran cscal implementation.
pub unsafe fn register_cscal(f: CscalFnPtr) {
    CSCAL
        .set(f)
        .expect("cscal already registered (can only be set once)");
}

/// Register the Fortran zscal function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran zscal implementation.
pub unsafe fn register_zscal(f: ZscalFnPtr) {
    ZSCAL
        .set(f)
        .expect("zscal already registered (can only be set once)");
}

/// Register the Fortran csscal function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran csscal implementation.
pub unsafe fn register_csscal(f: CsscalFnPtr) {
    CSSCAL
        .set(f)
        .expect("csscal already registered (can only be set once)");
}

/// Register the Fortran zdscal function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran zdscal implementation.
pub unsafe fn register_zdscal(f: ZdscalFnPtr) {
    ZDSCAL
        .set(f)
        .expect("zdscal already registered (can only be set once)");
}

// BLAS Level 3 registration

/// Register the Fortran dgemm function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dgemm implementation with the
/// correct calling convention and signature.
///
/// # Panics
///
/// Panics if dgemm has already been registered.
pub unsafe fn register_dgemm(f: DgemmFnPtr) {
    DGEMM
        .set(f)
        .expect("dgemm already registered (can only be set once)");
}

/// Register the Fortran sgemm function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran sgemm implementation.
///
/// # Panics
///
/// Panics if sgemm has already been registered.
pub unsafe fn register_sgemm(f: SgemmFnPtr) {
    SGEMM
        .set(f)
        .expect("sgemm already registered (can only be set once)");
}

/// Register the Fortran zgemm function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran zgemm implementation.
///
/// # Panics
///
/// Panics if zgemm has already been registered.
pub unsafe fn register_zgemm(f: ZgemmFnPtr) {
    ZGEMM
        .set(f)
        .expect("zgemm already registered (can only be set once)");
}

/// Register the Fortran cgemm function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran cgemm implementation.
///
/// # Panics
///
/// Panics if cgemm has already been registered.
pub unsafe fn register_cgemm(f: CgemmFnPtr) {
    CGEMM
        .set(f)
        .expect("cgemm already registered (can only be set once)");
}

/// Register the Fortran dsymm function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dsymm implementation.
pub unsafe fn register_dsymm(f: DsymmFnPtr) {
    DSYMM
        .set(f)
        .expect("dsymm already registered (can only be set once)");
}

/// Register the Fortran dsyrk function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dsyrk implementation.
pub unsafe fn register_dsyrk(f: DsyrkFnPtr) {
    DSYRK
        .set(f)
        .expect("dsyrk already registered (can only be set once)");
}

/// Register the Fortran dsyr2k function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dsyr2k implementation.
pub unsafe fn register_dsyr2k(f: Dsyr2kFnPtr) {
    DSYR2K
        .set(f)
        .expect("dsyr2k already registered (can only be set once)");
}

/// Register the Fortran dtrmm function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dtrmm implementation.
pub unsafe fn register_dtrmm(f: DtrmmFnPtr) {
    DTRMM
        .set(f)
        .expect("dtrmm already registered (can only be set once)");
}

/// Register the Fortran dtrsm function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dtrsm implementation.
pub unsafe fn register_dtrsm(f: DtrsmFnPtr) {
    DTRSM
        .set(f)
        .expect("dtrsm already registered (can only be set once)");
}

/// Register the Fortran srot function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran srot implementation.
pub unsafe fn register_srot(f: SrotFnPtr) {
    SROT.set(f)
        .expect("srot already registered (can only be set once)");
}

/// Register the Fortran drot function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran drot implementation.
pub unsafe fn register_drot(f: DrotFnPtr) {
    DROT.set(f)
        .expect("drot already registered (can only be set once)");
}

/// Register the Fortran srotg function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran srotg implementation.
pub unsafe fn register_srotg(f: SrotgFnPtr) {
    SROTG
        .set(f)
        .expect("srotg already registered (can only be set once)");
}

/// Register the Fortran drotg function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran drotg implementation.
pub unsafe fn register_drotg(f: DrotgFnPtr) {
    DROTG
        .set(f)
        .expect("drotg already registered (can only be set once)");
}

/// Register the Fortran srotm function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran srotm implementation.
pub unsafe fn register_srotm(f: SrotmFnPtr) {
    SROTM
        .set(f)
        .expect("srotm already registered (can only be set once)");
}

/// Register the Fortran drotm function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran drotm implementation.
pub unsafe fn register_drotm(f: DrotmFnPtr) {
    DROTM
        .set(f)
        .expect("drotm already registered (can only be set once)");
}

/// Register the Fortran srotmg function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran srotmg implementation.
pub unsafe fn register_srotmg(f: SrotmgFnPtr) {
    SROTMG
        .set(f)
        .expect("srotmg already registered (can only be set once)");
}

/// Register the Fortran drotmg function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran drotmg implementation.
pub unsafe fn register_drotmg(f: DrotmgFnPtr) {
    DROTMG
        .set(f)
        .expect("drotmg already registered (can only be set once)");
}

/// Register the Fortran scabs1 function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran scabs1 implementation.
pub unsafe fn register_scabs1(f: Scabs1FnPtr) {
    SCABS1
        .set(f)
        .expect("scabs1 already registered (can only be set once)");
}

/// Register the Fortran dcabs1 function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dcabs1 implementation.
pub unsafe fn register_dcabs1(f: Dcabs1FnPtr) {
    DCABS1
        .set(f)
        .expect("dcabs1 already registered (can only be set once)");
}

/// Register the Fortran sdot function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran sdot implementation.
pub unsafe fn register_sdot(f: SdotFnPtr) {
    SDOT.set(f)
        .expect("sdot already registered (can only be set once)");
}

/// Register the Fortran ddot function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran ddot implementation.
pub unsafe fn register_ddot(f: DdotFnPtr) {
    DDOT.set(f)
        .expect("ddot already registered (can only be set once)");
}

/// Register the Fortran cdotu function pointer (return value convention).
///
/// # Safety
///
/// The function pointer must be a valid Fortran cdotu implementation
/// using the return value convention.
pub unsafe fn register_cdotu(f: CdotuFnPtr) {
    CDOTU_PTR
        .set(FnPtrWrapper(f as *const ()))
        .expect("cdotu already registered (can only be set once)");
}

/// Register a raw cdotu function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran cdotu implementation.
/// The calling convention must match the currently set `ComplexReturnStyle`.
pub unsafe fn register_cdotu_raw(ptr: *const ()) {
    CDOTU_PTR
        .set(FnPtrWrapper(ptr))
        .expect("cdotu already registered (can only be set once)");
}

/// Register the Fortran zdotu function pointer (return value convention).
///
/// # Safety
///
/// The function pointer must be a valid Fortran zdotu implementation
/// using the return value convention.
pub unsafe fn register_zdotu(f: ZdotuFnPtr) {
    ZDOTU_PTR
        .set(FnPtrWrapper(f as *const ()))
        .expect("zdotu already registered (can only be set once)");
}

/// Register a raw zdotu function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran zdotu implementation.
/// The calling convention must match the currently set `ComplexReturnStyle`.
pub unsafe fn register_zdotu_raw(ptr: *const ()) {
    ZDOTU_PTR
        .set(FnPtrWrapper(ptr))
        .expect("zdotu already registered (can only be set once)");
}

/// Register the Fortran cdotc function pointer (return value convention).
///
/// # Safety
///
/// The function pointer must be a valid Fortran cdotc implementation
/// using the return value convention.
pub unsafe fn register_cdotc(f: CdotcFnPtr) {
    CDOTC_PTR
        .set(FnPtrWrapper(f as *const ()))
        .expect("cdotc already registered (can only be set once)");
}

/// Register a raw cdotc function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran cdotc implementation.
/// The calling convention must match the currently set `ComplexReturnStyle`.
pub unsafe fn register_cdotc_raw(ptr: *const ()) {
    CDOTC_PTR
        .set(FnPtrWrapper(ptr))
        .expect("cdotc already registered (can only be set once)");
}

/// Register the Fortran zdotc function pointer (return value convention).
///
/// # Safety
///
/// The function pointer must be a valid Fortran zdotc implementation
/// using the return value convention.
pub unsafe fn register_zdotc(f: ZdotcFnPtr) {
    ZDOTC_PTR
        .set(FnPtrWrapper(f as *const ()))
        .expect("zdotc already registered (can only be set once)");
}

/// Register a raw zdotc function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran zdotc implementation.
/// The calling convention must match the currently set `ComplexReturnStyle`.
pub unsafe fn register_zdotc_raw(ptr: *const ()) {
    ZDOTC_PTR
        .set(FnPtrWrapper(ptr))
        .expect("zdotc already registered (can only be set once)");
}

/// Register the Fortran sdsdot function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran sdsdot implementation.
pub unsafe fn register_sdsdot(f: SdsdotFnPtr) {
    SDSDOT
        .set(f)
        .expect("sdsdot already registered (can only be set once)");
}

/// Register the Fortran dsdot function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dsdot implementation.
pub unsafe fn register_dsdot(f: DsdotFnPtr) {
    DSDOT
        .set(f)
        .expect("dsdot already registered (can only be set once)");
}

/// Register the Fortran snrm2 function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran snrm2 implementation.
pub unsafe fn register_snrm2(f: Snrm2FnPtr) {
    SNRM2
        .set(f)
        .expect("snrm2 already registered (can only be set once)");
}

/// Register the Fortran dnrm2 function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dnrm2 implementation.
pub unsafe fn register_dnrm2(f: Dnrm2FnPtr) {
    DNRM2
        .set(f)
        .expect("dnrm2 already registered (can only be set once)");
}

/// Register the Fortran scnrm2 function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran scnrm2 implementation.
pub unsafe fn register_scnrm2(f: Scnrm2FnPtr) {
    SCNRM2
        .set(f)
        .expect("scnrm2 already registered (can only be set once)");
}

/// Register the Fortran dznrm2 function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dznrm2 implementation.
pub unsafe fn register_dznrm2(f: Dznrm2FnPtr) {
    DZNRM2
        .set(f)
        .expect("dznrm2 already registered (can only be set once)");
}

/// Register the Fortran sasum function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran sasum implementation.
pub unsafe fn register_sasum(f: SasumFnPtr) {
    SASUM
        .set(f)
        .expect("sasum already registered (can only be set once)");
}

/// Register the Fortran dasum function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dasum implementation.
pub unsafe fn register_dasum(f: DasumFnPtr) {
    DASUM
        .set(f)
        .expect("dasum already registered (can only be set once)");
}

/// Register the Fortran scasum function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran scasum implementation.
pub unsafe fn register_scasum(f: ScasumFnPtr) {
    SCASUM
        .set(f)
        .expect("scasum already registered (can only be set once)");
}

/// Register the Fortran dzasum function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran dzasum implementation.
pub unsafe fn register_dzasum(f: DzasumFnPtr) {
    DZASUM
        .set(f)
        .expect("dzasum already registered (can only be set once)");
}

/// Register the Fortran isamax function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran isamax implementation.
pub unsafe fn register_isamax(f: IsamaxFnPtr) {
    ISAMAX
        .set(f)
        .expect("isamax already registered (can only be set once)");
}

/// Register the Fortran idamax function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran idamax implementation.
pub unsafe fn register_idamax(f: IdamaxFnPtr) {
    IDAMAX
        .set(f)
        .expect("idamax already registered (can only be set once)");
}

/// Register the Fortran icamax function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran icamax implementation.
pub unsafe fn register_icamax(f: IcamaxFnPtr) {
    ICAMAX
        .set(f)
        .expect("icamax already registered (can only be set once)");
}

/// Register the Fortran izamax function pointer.
///
/// # Safety
///
/// The function pointer must be a valid Fortran izamax implementation.
pub unsafe fn register_izamax(f: IzamaxFnPtr) {
    IZAMAX
        .set(f)
        .expect("izamax already registered (can only be set once)");
}

// =============================================================================
// Internal getters (used by blas3/gemm.rs etc.)
// =============================================================================

#[inline]
pub(crate) fn get_dgemm() -> DgemmFnPtr {
    *DGEMM
        .get()
        .expect("dgemm not registered: call register_dgemm() first")
}

#[inline]
pub(crate) fn get_sgemm() -> SgemmFnPtr {
    *SGEMM
        .get()
        .expect("sgemm not registered: call register_sgemm() first")
}

#[inline]
pub(crate) fn get_zgemm() -> ZgemmFnPtr {
    *ZGEMM
        .get()
        .expect("zgemm not registered: call register_zgemm() first")
}

#[inline]
pub(crate) fn get_cgemm() -> CgemmFnPtr {
    *CGEMM
        .get()
        .expect("cgemm not registered: call register_cgemm() first")
}

#[inline]
pub(crate) fn get_dsymm() -> DsymmFnPtr {
    *DSYMM
        .get()
        .expect("dsymm not registered: call register_dsymm() first")
}

#[inline]
pub(crate) fn get_dsyrk() -> DsyrkFnPtr {
    *DSYRK
        .get()
        .expect("dsyrk not registered: call register_dsyrk() first")
}

#[inline]
pub(crate) fn get_dsyr2k() -> Dsyr2kFnPtr {
    *DSYR2K
        .get()
        .expect("dsyr2k not registered: call register_dsyr2k() first")
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
    CDOTU_PTR
        .get()
        .expect("cdotu not registered: call register_cdotu() first")
        .0
}

#[inline]
pub(crate) fn get_zdotu_ptr() -> *const () {
    ZDOTU_PTR
        .get()
        .expect("zdotu not registered: call register_zdotu() first")
        .0
}

#[inline]
pub(crate) fn get_cdotc_ptr() -> *const () {
    CDOTC_PTR
        .get()
        .expect("cdotc not registered: call register_cdotc() first")
        .0
}

#[inline]
pub(crate) fn get_zdotc_ptr() -> *const () {
    ZDOTC_PTR
        .get()
        .expect("zdotc not registered: call register_zdotc() first")
        .0
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
