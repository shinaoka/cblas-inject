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
// Function pointer storage (OnceLock per function)
// =============================================================================

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
