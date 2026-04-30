#[cfg(not(feature = "openblas"))]
use std::ffi::c_char;
use std::ffi::c_void;
use std::ptr;

use cblas_inject::{
    blasint, CBLAS_INJECT_STATUS_ALREADY_REGISTERED, CBLAS_INJECT_STATUS_NULL_POINTER,
    CBLAS_INJECT_STATUS_OK,
};
#[cfg(not(feature = "openblas"))]
use cblas_inject::{register_dgemm, register_zgemm, BlasInt32, BlasInt64};
#[cfg(not(feature = "openblas"))]
use num_complex::Complex64;

extern "C" {
    fn cblas_inject_register_dgemm_lp64(f: *const c_void) -> i32;
    fn cblas_inject_register_dgemm_ilp64(f: *const c_void) -> i32;
    fn cblas_inject_register_zgemm_lp64(f: *const c_void) -> i32;
    fn cblas_inject_register_zgemm_ilp64(f: *const c_void) -> i32;
    fn cblas_inject_blas_int_width() -> i32;
    fn cblas_inject_supports_lp64_registration() -> i32;
    fn cblas_inject_supports_ilp64_registration() -> i32;
}

#[cfg(not(feature = "openblas"))]
unsafe extern "C" fn mock_dgemm_current(
    _transa: *const c_char,
    _transb: *const c_char,
    _m: *const blasint,
    _n: *const blasint,
    _k: *const blasint,
    _alpha: *const f64,
    _a: *const f64,
    _lda: *const blasint,
    _b: *const f64,
    _ldb: *const blasint,
    _beta: *const f64,
    _c: *mut f64,
    _ldc: *const blasint,
) {
}

#[cfg(not(feature = "openblas"))]
unsafe extern "C" fn mock_zgemm_current(
    _transa: *const c_char,
    _transb: *const c_char,
    _m: *const blasint,
    _n: *const blasint,
    _k: *const blasint,
    _alpha: *const Complex64,
    _a: *const Complex64,
    _lda: *const blasint,
    _b: *const Complex64,
    _ldb: *const blasint,
    _beta: *const Complex64,
    _c: *mut Complex64,
    _ldc: *const blasint,
) {
}

#[cfg(not(feature = "openblas"))]
unsafe extern "C" fn mock_dgemm_lp64(
    _transa: *const c_char,
    _transb: *const c_char,
    _m: *const BlasInt32,
    _n: *const BlasInt32,
    _k: *const BlasInt32,
    _alpha: *const f64,
    _a: *const f64,
    _lda: *const BlasInt32,
    _b: *const f64,
    _ldb: *const BlasInt32,
    _beta: *const f64,
    _c: *mut f64,
    _ldc: *const BlasInt32,
) {
}

#[cfg(not(feature = "openblas"))]
unsafe extern "C" fn mock_dgemm_ilp64(
    _transa: *const c_char,
    _transb: *const c_char,
    _m: *const BlasInt64,
    _n: *const BlasInt64,
    _k: *const BlasInt64,
    _alpha: *const f64,
    _a: *const f64,
    _lda: *const BlasInt64,
    _b: *const f64,
    _ldb: *const BlasInt64,
    _beta: *const f64,
    _c: *mut f64,
    _ldc: *const BlasInt64,
) {
}

#[cfg(not(feature = "openblas"))]
unsafe extern "C" fn mock_zgemm_lp64(
    _transa: *const c_char,
    _transb: *const c_char,
    _m: *const BlasInt32,
    _n: *const BlasInt32,
    _k: *const BlasInt32,
    _alpha: *const Complex64,
    _a: *const Complex64,
    _lda: *const BlasInt32,
    _b: *const Complex64,
    _ldb: *const BlasInt32,
    _beta: *const Complex64,
    _c: *mut Complex64,
    _ldc: *const BlasInt32,
) {
}

#[cfg(not(feature = "openblas"))]
unsafe extern "C" fn mock_zgemm_ilp64(
    _transa: *const c_char,
    _transb: *const c_char,
    _m: *const BlasInt64,
    _n: *const BlasInt64,
    _k: *const BlasInt64,
    _alpha: *const Complex64,
    _a: *const Complex64,
    _lda: *const BlasInt64,
    _b: *const Complex64,
    _ldb: *const BlasInt64,
    _beta: *const Complex64,
    _c: *mut Complex64,
    _ldc: *const BlasInt64,
) {
}

#[test]
fn c_registration_api_reports_capabilities_and_rejects_null_pointers() {
    assert_eq!(CBLAS_INJECT_STATUS_OK, 0);
    assert_eq!(CBLAS_INJECT_STATUS_NULL_POINTER, 1);
    assert_eq!(CBLAS_INJECT_STATUS_ALREADY_REGISTERED, 2);
    unsafe {
        assert_eq!(
            cblas_inject_blas_int_width(),
            (std::mem::size_of::<blasint>() * 8) as i32
        );
        assert_eq!(cblas_inject_supports_lp64_registration(), 1);
        assert_eq!(cblas_inject_supports_ilp64_registration(), 1);
    }

    unsafe {
        assert_eq!(
            cblas_inject_register_dgemm_lp64(ptr::null()),
            CBLAS_INJECT_STATUS_NULL_POINTER
        );
        assert_eq!(
            cblas_inject_register_dgemm_ilp64(ptr::null()),
            CBLAS_INJECT_STATUS_NULL_POINTER
        );
        assert_eq!(
            cblas_inject_register_zgemm_lp64(ptr::null()),
            CBLAS_INJECT_STATUS_NULL_POINTER
        );
        assert_eq!(
            cblas_inject_register_zgemm_ilp64(ptr::null()),
            CBLAS_INJECT_STATUS_NULL_POINTER
        );
    }
}

#[cfg(not(feature = "openblas"))]
#[test]
fn c_registration_api_reports_duplicate_for_legacy_current_width() {
    unsafe {
        register_dgemm(mock_dgemm_current);
        register_zgemm(mock_zgemm_current);
    }

    #[cfg(not(feature = "ilp64"))]
    unsafe {
        assert_eq!(
            cblas_inject_register_dgemm_lp64(mock_dgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject_register_zgemm_lp64(mock_zgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject_register_dgemm_ilp64(mock_dgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject_register_zgemm_ilp64(mock_zgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
    }

    #[cfg(feature = "ilp64")]
    unsafe {
        assert_eq!(
            cblas_inject_register_dgemm_ilp64(mock_dgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject_register_zgemm_ilp64(mock_zgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject_register_dgemm_lp64(mock_dgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject_register_zgemm_lp64(mock_zgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
    }

    unsafe {
        assert_eq!(
            cblas_inject_register_dgemm_lp64(mock_dgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject_register_dgemm_ilp64(mock_dgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject_register_zgemm_lp64(mock_zgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject_register_zgemm_ilp64(mock_zgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
    }
}
