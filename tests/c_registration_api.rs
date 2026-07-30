#[cfg(not(feature = "openblas"))]
use std::ffi::c_char;
use std::ffi::c_void;
use std::ptr;

#[cfg(not(feature = "openblas"))]
use cblas_inject::{register_dgemm, register_zgemm, BlasInt32, BlasInt64};
use cblas_inject::{
    CBLAS_INJECT_STATUS_ALREADY_REGISTERED, CBLAS_INJECT_STATUS_NULL_POINTER,
    CBLAS_INJECT_STATUS_OK,
};
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

trait Scalar: Copy + PartialEq + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> {
    fn zero() -> Self;
    fn conj(self) -> Self;
}
impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn conj(self) -> Self {
        self
    }
}
impl Scalar for Complex64 {
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn conj(self) -> Self {
        Complex64::new(self.re, -self.im)
    }
}

unsafe fn gemm<T: Scalar, I: Copy + Into<i64>>(
    transa: *const c_char,
    transb: *const c_char,
    m: *const I,
    n: *const I,
    k: *const I,
    alpha: *const T,
    a: *const T,
    lda: *const I,
    b: *const T,
    ldb: *const I,
    beta: *const T,
    c: *mut T,
    ldc: *const I,
) {
    let (m, n, k, lda, ldb, ldc) = (
        (*m).into() as usize,
        (*n).into() as usize,
        (*k).into() as usize,
        (*lda).into() as usize,
        (*ldb).into() as usize,
        (*ldc).into() as usize,
    );
    let ta = *transa as u8;
    let tb = *transb as u8;
    for j in 0..n {
        for i in 0..m {
            let mut sum = T::zero();
            for p in 0..k {
                let ai = if ta == b'N' { i + p * lda } else { p + i * lda };
                let bi = if tb == b'N' { p + j * ldb } else { j + p * ldb };
                let av = *a.add(ai);
                let bv = *b.add(bi);
                sum = sum
                    + if ta == b'C' { av.conj() } else { av }
                        * if tb == b'C' { bv.conj() } else { bv };
            }
            let out = c.add(i + j * ldc);
            *out = *alpha * sum
                + if *beta == T::zero() {
                    T::zero()
                } else {
                    *beta * *out
                };
        }
    }
}

macro_rules! callback {
    ($name:ident, $ty:ty, $int:ty) => {
        unsafe extern "C" fn $name(
            t1: *const c_char,
            t2: *const c_char,
            m: *const $int,
            n: *const $int,
            k: *const $int,
            alpha: *const $ty,
            a: *const $ty,
            lda: *const $int,
            b: *const $ty,
            ldb: *const $int,
            beta: *const $ty,
            c: *mut $ty,
            ldc: *const $int,
        ) {
            unsafe { gemm(t1, t2, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }
        }
    };
}
callback!(mock_dgemm_current, f64, BlasInt32);
callback!(mock_dgemm_lp64, f64, BlasInt32);
callback!(mock_dgemm_ilp64, f64, BlasInt64);
callback!(mock_zgemm_current, Complex64, BlasInt32);
callback!(mock_zgemm_lp64, Complex64, BlasInt32);
callback!(mock_zgemm_ilp64, Complex64, BlasInt64);

#[test]
fn c_registration_api_reports_capabilities_and_rejects_null_pointers() {
    assert_eq!(CBLAS_INJECT_STATUS_OK, 0);
    assert_eq!(CBLAS_INJECT_STATUS_NULL_POINTER, 1);
    assert_eq!(CBLAS_INJECT_STATUS_ALREADY_REGISTERED, 2);
    unsafe {
        assert_eq!(cblas_inject_blas_int_width(), 32);
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

        let alpha = Complex64::new(2.0, 0.0);
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, 4.0);
        let beta = Complex64::new(0.0, 0.0);
        let mut c = Complex64::new(f64::NAN, f64::NAN);
        let one = 1i32;
        mock_zgemm_current(
            b"C".as_ptr() as *const c_char,
            b"N".as_ptr() as *const c_char,
            &one,
            &one,
            &one,
            &alpha,
            &a,
            &one,
            &b,
            &one,
            &beta,
            &mut c,
            &one,
        );
        assert_eq!(c, Complex64::new(22.0, -4.0));
    }

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
