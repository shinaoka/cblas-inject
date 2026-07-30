#![cfg(not(feature = "openblas"))]

use cblas_inject::{register_cgemm, register_sgemm, BlasInt32, BlasInt64};
use cblas_inject::{
    CBLAS_INJECT_STATUS_ALREADY_REGISTERED, CBLAS_INJECT_STATUS_NULL_POINTER,
    CBLAS_INJECT_STATUS_OK,
};
use num_complex::Complex32;
use std::ffi::{c_char, c_void};
use std::ptr;

extern "C" {
    fn cblas_inject_register_sgemm_lp64(f: *const c_void) -> i32;
    fn cblas_inject_register_sgemm_ilp64(f: *const c_void) -> i32;
    fn cblas_inject_register_cgemm_lp64(f: *const c_void) -> i32;
    fn cblas_inject_register_cgemm_ilp64(f: *const c_void) -> i32;
}

trait Scalar: Copy + PartialEq + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> {
    fn zero() -> Self;
    fn conj(self) -> Self;
}
impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn conj(self) -> Self {
        self
    }
}
impl Scalar for Complex32 {
    fn zero() -> Self {
        Complex32::new(0.0, 0.0)
    }
    fn conj(self) -> Self {
        Complex32::new(self.re, -self.im)
    }
}

unsafe fn gemm<T: Scalar, I: Copy + Into<i64>>(
    ta: *const c_char,
    tb: *const c_char,
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
    for j in 0..n {
        for i in 0..m {
            let mut sum = T::zero();
            for p in 0..k {
                let ai = if *ta as u8 == b'N' {
                    i + p * lda
                } else {
                    p + i * lda
                };
                let bi = if *tb as u8 == b'N' {
                    p + j * ldb
                } else {
                    j + p * ldb
                };
                let av = *a.add(ai);
                let bv = *b.add(bi);
                sum = sum
                    + if *ta as u8 == b'C' { av.conj() } else { av }
                        * if *tb as u8 == b'C' { bv.conj() } else { bv };
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
    ($name:ident,$ty:ty,$int:ty) => {
        unsafe extern "C" fn $name(
            ta: *const c_char,
            tb: *const c_char,
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
            unsafe { gemm(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) }
        }
    };
}
callback!(sgemm_lp64, f32, BlasInt32);
callback!(sgemm_ilp64, f32, BlasInt64);
callback!(cgemm_lp64, Complex32, BlasInt32);
callback!(cgemm_ilp64, Complex32, BlasInt64);

#[test]
fn s_and_c_registration_reject_null_and_report_duplicates() {
    unsafe {
        assert_eq!(
            cblas_inject_register_sgemm_lp64(ptr::null()),
            CBLAS_INJECT_STATUS_NULL_POINTER
        );
        assert_eq!(
            cblas_inject_register_sgemm_ilp64(ptr::null()),
            CBLAS_INJECT_STATUS_NULL_POINTER
        );
        assert_eq!(
            cblas_inject_register_cgemm_lp64(ptr::null()),
            CBLAS_INJECT_STATUS_NULL_POINTER
        );
        assert_eq!(
            cblas_inject_register_cgemm_ilp64(ptr::null()),
            CBLAS_INJECT_STATUS_NULL_POINTER
        );
        register_sgemm(sgemm_lp64);
        register_cgemm(cgemm_lp64);
        assert_eq!(
            cblas_inject_register_sgemm_lp64(sgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject_register_cgemm_lp64(cgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject_register_sgemm_ilp64(sgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject_register_cgemm_ilp64(cgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject_register_sgemm_ilp64(sgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
        assert_eq!(
            cblas_inject_register_cgemm_ilp64(cgemm_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_ALREADY_REGISTERED
        );
    }
}
