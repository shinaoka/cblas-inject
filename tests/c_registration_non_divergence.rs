#![cfg(not(feature = "openblas"))]

use std::ffi::{c_char, c_void};

use cblas_inject::{
    is_dgemm_registered, is_zgemm_registered, BlasInt32, BlasInt64,
    CBLAS_INJECT_STATUS_ALREADY_REGISTERED, CBLAS_INJECT_STATUS_OK,
};
use num_complex::Complex64;

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

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_c_registration_populates_legacy_current_abi_storage() {
    unsafe {
        assert_eq!(
            cblas_inject::cblas_inject_register_dgemm_lp64(mock_dgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zgemm_lp64(mock_zgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
    }

    assert!(is_dgemm_registered());
    assert!(is_zgemm_registered());

    unsafe {
        assert_eq!(
            cblas_inject::cblas_inject_register_dgemm_lp64(mock_dgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zgemm_lp64(mock_zgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_dgemm_ilp64(mock_dgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zgemm_ilp64(mock_zgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
    }
}

#[cfg(feature = "ilp64")]
#[test]
fn ilp64_c_registration_populates_legacy_current_abi_storage() {
    unsafe {
        assert_eq!(
            cblas_inject::cblas_inject_register_dgemm_ilp64(mock_dgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zgemm_ilp64(mock_zgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
    }

    assert!(is_dgemm_registered());
    assert!(is_zgemm_registered());

    unsafe {
        assert_eq!(
            cblas_inject::cblas_inject_register_dgemm_ilp64(mock_dgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zgemm_ilp64(mock_zgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_dgemm_lp64(mock_dgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zgemm_lp64(mock_zgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
    }
}
