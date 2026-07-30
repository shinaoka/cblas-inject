#![cfg(not(feature = "openblas"))]

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};

#[cfg(feature = "ilp64")]
use cblas_inject::BlasInt32;
#[cfg(not(feature = "ilp64"))]
use cblas_inject::BlasInt64;
#[cfg(feature = "ilp64")]
use cblas_inject::{cblas_cgemm_64, cblas_dgemm_64, cblas_sgemm_64, cblas_zgemm_64};
use cblas_inject::{
    cblas_dgemm, cblas_zgemm, CblasColMajor, CblasNoTrans, CblasRowMajor, CBLAS_INJECT_STATUS_OK,
};
#[cfg(feature = "ilp64")]
use num_complex::Complex32;
use num_complex::Complex64;

static DGEMM_M: AtomicI64 = AtomicI64::new(0);
static DGEMM_N: AtomicI64 = AtomicI64::new(0);
static DGEMM_K: AtomicI64 = AtomicI64::new(0);
static DGEMM_LDA: AtomicI64 = AtomicI64::new(0);
static DGEMM_LDB: AtomicI64 = AtomicI64::new(0);
static DGEMM_LDC: AtomicI64 = AtomicI64::new(0);
static DGEMM_CALLS: AtomicUsize = AtomicUsize::new(0);

static ZGEMM_M: AtomicI64 = AtomicI64::new(0);
static ZGEMM_N: AtomicI64 = AtomicI64::new(0);
static ZGEMM_K: AtomicI64 = AtomicI64::new(0);
static ZGEMM_LDA: AtomicI64 = AtomicI64::new(0);
static ZGEMM_LDB: AtomicI64 = AtomicI64::new(0);
static ZGEMM_LDC: AtomicI64 = AtomicI64::new(0);
static ZGEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
static ZGEMM_ALPHA_RE: AtomicU64 = AtomicU64::new(0);
static ZGEMM_ALPHA_IM: AtomicU64 = AtomicU64::new(0);
static ZGEMM_BETA_RE: AtomicU64 = AtomicU64::new(0);
static ZGEMM_BETA_IM: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "ilp64")]
static SGEMM_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "ilp64")]
static CGEMM_CALLS: AtomicUsize = AtomicUsize::new(0);

unsafe fn complex_gemm<I: Copy + Into<i64>>(
    ta: *const c_char,
    tb: *const c_char,
    m: *const I,
    n: *const I,
    k: *const I,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const I,
    b: *const Complex64,
    ldb: *const I,
    beta: *const Complex64,
    c: *mut Complex64,
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
    let load = |p: *const Complex64, i: usize, trans: u8| unsafe {
        let value = *p.add(i);
        if trans == b'C' {
            value.conj()
        } else {
            value
        }
    };
    for j in 0..n {
        for i in 0..m {
            let mut sum = Complex64::new(0.0, 0.0);
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
                sum += load(a, ai, *ta as u8) * load(b, bi, *tb as u8);
            }
            let out = c.add(i + j * ldc);
            *out = *alpha * sum
                + if *beta == Complex64::new(0.0, 0.0) {
                    Complex64::new(0.0, 0.0)
                } else {
                    *beta * *out
                };
        }
    }
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_dgemm_lp64(
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
) {
    DGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    DGEMM_M.store(i64::from(unsafe { *m }), Ordering::SeqCst);
    DGEMM_N.store(i64::from(unsafe { *n }), Ordering::SeqCst);
    DGEMM_K.store(i64::from(unsafe { *k }), Ordering::SeqCst);
    DGEMM_LDA.store(i64::from(unsafe { *lda }), Ordering::SeqCst);
    DGEMM_LDB.store(i64::from(unsafe { *ldb }), Ordering::SeqCst);
    DGEMM_LDC.store(i64::from(unsafe { *ldc }), Ordering::SeqCst);
    let (m, n, k) = (
        unsafe { *m } as usize,
        unsafe { *n } as usize,
        unsafe { *k } as usize,
    );
    let (lda, ldb, ldc) = (
        unsafe { *lda } as usize,
        unsafe { *ldb } as usize,
        unsafe { *ldc } as usize,
    );
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                let ai = if unsafe { *transa as u8 } == b'N' {
                    i + p * lda
                } else {
                    p + i * lda
                };
                let bi = if unsafe { *transb as u8 } == b'N' {
                    p + j * ldb
                } else {
                    j + p * ldb
                };
                sum += unsafe { *a.add(ai) * *b.add(bi) };
            }
            let out = unsafe { c.add(i + j * ldc) };
            unsafe {
                *out = *alpha * sum + if *beta == 0.0 { 0.0 } else { *beta * *out };
            }
        }
    }
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn mock_dgemm_ilp64(
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
) {
    DGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    DGEMM_M.store(unsafe { *m }, Ordering::SeqCst);
    DGEMM_N.store(unsafe { *n }, Ordering::SeqCst);
    DGEMM_K.store(unsafe { *k }, Ordering::SeqCst);
    DGEMM_LDA.store(unsafe { *lda }, Ordering::SeqCst);
    DGEMM_LDB.store(unsafe { *ldb }, Ordering::SeqCst);
    DGEMM_LDC.store(unsafe { *ldc }, Ordering::SeqCst);
    let (m, n, k) = (
        unsafe { *m } as usize,
        unsafe { *n } as usize,
        unsafe { *k } as usize,
    );
    let (lda, ldb, ldc) = (
        unsafe { *lda } as usize,
        unsafe { *ldb } as usize,
        unsafe { *ldc } as usize,
    );
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                let ai = if unsafe { *transa as u8 } == b'N' {
                    i + p * lda
                } else {
                    p + i * lda
                };
                let bi = if unsafe { *transb as u8 } == b'N' {
                    p + j * ldb
                } else {
                    j + p * ldb
                };
                sum += unsafe { *a.add(ai) * *b.add(bi) };
            }
            let out = unsafe { c.add(i + j * ldc) };
            unsafe {
                *out = *alpha * sum + if *beta == 0.0 { 0.0 } else { *beta * *out };
            }
        }
    }
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_zgemm_lp64(
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
) {
    ZGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    ZGEMM_M.store(i64::from(unsafe { *m }), Ordering::SeqCst);
    ZGEMM_N.store(i64::from(unsafe { *n }), Ordering::SeqCst);
    ZGEMM_K.store(i64::from(unsafe { *k }), Ordering::SeqCst);
    ZGEMM_LDA.store(i64::from(unsafe { *lda }), Ordering::SeqCst);
    ZGEMM_LDB.store(i64::from(unsafe { *ldb }), Ordering::SeqCst);
    ZGEMM_LDC.store(i64::from(unsafe { *ldc }), Ordering::SeqCst);
    ZGEMM_ALPHA_RE.store(unsafe { (*alpha).re.to_bits() }, Ordering::SeqCst);
    ZGEMM_ALPHA_IM.store(unsafe { (*alpha).im.to_bits() }, Ordering::SeqCst);
    ZGEMM_BETA_RE.store(unsafe { (*beta).re.to_bits() }, Ordering::SeqCst);
    ZGEMM_BETA_IM.store(unsafe { (*beta).im.to_bits() }, Ordering::SeqCst);
    unsafe {
        complex_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_sgemm_lp64(
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
) {
    SGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    let (m, n, k, lda, ldb, ldc) = unsafe {
        (
            *m as usize,
            *n as usize,
            *k as usize,
            *lda as usize,
            *ldb as usize,
            *ldc as usize,
        )
    };
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                let ai = if unsafe { *transa as u8 } == b'N' {
                    i + p * lda
                } else {
                    p + i * lda
                };
                let bi = if unsafe { *transb as u8 } == b'N' {
                    p + j * ldb
                } else {
                    j + p * ldb
                };
                sum += unsafe { *a.add(ai) * *b.add(bi) };
            }
            let out = unsafe { c.add(i + j * ldc) };
            unsafe {
                *out = *alpha * sum + if *beta == 0.0 { 0.0 } else { *beta * *out };
            }
        }
    }
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_cgemm_lp64(
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
) {
    CGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    let (m, n, k, lda, ldb, ldc) = unsafe {
        (
            *m as usize,
            *n as usize,
            *k as usize,
            *lda as usize,
            *ldb as usize,
            *ldc as usize,
        )
    };
    for j in 0..n {
        for i in 0..m {
            let mut sum = Complex32::new(0.0, 0.0);
            for p in 0..k {
                let ai = if unsafe { *transa as u8 } == b'N' {
                    i + p * lda
                } else {
                    p + i * lda
                };
                let bi = if unsafe { *transb as u8 } == b'N' {
                    p + j * ldb
                } else {
                    j + p * ldb
                };
                let av = unsafe { *a.add(ai) };
                let bv = unsafe { *b.add(bi) };
                sum += if unsafe { *transa as u8 } == b'C' {
                    av.conj()
                } else {
                    av
                } * if unsafe { *transb as u8 } == b'C' {
                    bv.conj()
                } else {
                    bv
                };
            }
            let out = unsafe { c.add(i + j * ldc) };
            unsafe {
                *out = *alpha * sum
                    + if *beta == Complex32::new(0.0, 0.0) {
                        Complex32::new(0.0, 0.0)
                    } else {
                        *beta * *out
                    };
            }
        }
    }
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn mock_zgemm_ilp64(
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
) {
    ZGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    ZGEMM_M.store(unsafe { *m }, Ordering::SeqCst);
    ZGEMM_N.store(unsafe { *n }, Ordering::SeqCst);
    ZGEMM_K.store(unsafe { *k }, Ordering::SeqCst);
    ZGEMM_LDA.store(unsafe { *lda }, Ordering::SeqCst);
    ZGEMM_LDB.store(unsafe { *ldb }, Ordering::SeqCst);
    ZGEMM_LDC.store(unsafe { *ldc }, Ordering::SeqCst);
    ZGEMM_ALPHA_RE.store(unsafe { (*alpha).re.to_bits() }, Ordering::SeqCst);
    ZGEMM_ALPHA_IM.store(unsafe { (*alpha).im.to_bits() }, Ordering::SeqCst);
    ZGEMM_BETA_RE.store(unsafe { (*beta).re.to_bits() }, Ordering::SeqCst);
    ZGEMM_BETA_IM.store(unsafe { (*beta).im.to_bits() }, Ordering::SeqCst);
    unsafe {
        complex_gemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_cblas_dgemm_and_zgemm_dispatch_to_ilp64_fallback_provider() {
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

    let a = [1.0; 12];
    let b = [2.0; 20];
    let mut c = [0.0; 6];
    unsafe {
        cblas_dgemm(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            2,
            3,
            4,
            1.5,
            a.as_ptr(),
            2,
            b.as_ptr(),
            4,
            0.5,
            c.as_mut_ptr(),
            2,
        );
    }

    assert_eq!(c[0], 12.0);
    assert_eq!(DGEMM_M.load(Ordering::SeqCst), 2);
    assert_eq!(DGEMM_N.load(Ordering::SeqCst), 3);
    assert_eq!(DGEMM_K.load(Ordering::SeqCst), 4);
    assert_eq!(DGEMM_LDA.load(Ordering::SeqCst), 2);
    assert_eq!(DGEMM_LDB.load(Ordering::SeqCst), 4);
    assert_eq!(DGEMM_LDC.load(Ordering::SeqCst), 2);

    let alpha = Complex64::new(2.0, 3.0);
    let beta = Complex64::new(5.0, 7.0);
    let za = [Complex64::new(1.0, 0.0); 12];
    let zb = [Complex64::new(2.0, 0.0); 20];
    let mut zc = [Complex64::new(0.0, 0.0); 6];
    unsafe {
        cblas_zgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            2,
            3,
            4,
            &alpha,
            za.as_ptr(),
            4,
            zb.as_ptr(),
            3,
            &beta,
            zc.as_mut_ptr(),
            3,
        );
    }

    assert_eq!(zc[0], Complex64::new(16.0, 24.0));
    assert_eq!(ZGEMM_M.load(Ordering::SeqCst), 3);
    assert_eq!(ZGEMM_N.load(Ordering::SeqCst), 2);
    assert_eq!(ZGEMM_K.load(Ordering::SeqCst), 4);
    assert_eq!(ZGEMM_LDA.load(Ordering::SeqCst), 3);
    assert_eq!(ZGEMM_LDB.load(Ordering::SeqCst), 4);
    assert_eq!(ZGEMM_LDC.load(Ordering::SeqCst), 3);
    assert_eq!(f64::from_bits(ZGEMM_ALPHA_RE.load(Ordering::SeqCst)), 2.0);
    assert_eq!(f64::from_bits(ZGEMM_ALPHA_IM.load(Ordering::SeqCst)), 3.0);
    assert_eq!(f64::from_bits(ZGEMM_BETA_RE.load(Ordering::SeqCst)), 5.0);
    assert_eq!(f64::from_bits(ZGEMM_BETA_IM.load(Ordering::SeqCst)), 7.0);
}

#[cfg(feature = "ilp64")]
#[test]
fn ilp64_cblas_dgemm_and_zgemm_dispatch_to_lp64_fallback_provider() {
    unsafe {
        assert_eq!(
            cblas_inject::cblas_inject_register_dgemm_lp64(mock_dgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zgemm_lp64(mock_zgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_sgemm_lp64(mock_sgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_cgemm_lp64(mock_cgemm_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
    }

    let a = [1.0; 12];
    let b = [2.0; 20];
    let mut c = [0.0; 6];
    unsafe {
        cblas_dgemm(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            2,
            3,
            4,
            1.5,
            a.as_ptr(),
            2,
            b.as_ptr(),
            4,
            0.5,
            c.as_mut_ptr(),
            2,
        );
    }

    assert_eq!(c[0], 12.0);
    assert_eq!(DGEMM_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(DGEMM_M.load(Ordering::SeqCst), 2);
    assert_eq!(DGEMM_N.load(Ordering::SeqCst), 3);
    assert_eq!(DGEMM_K.load(Ordering::SeqCst), 4);
    assert_eq!(DGEMM_LDA.load(Ordering::SeqCst), 2);
    assert_eq!(DGEMM_LDB.load(Ordering::SeqCst), 4);
    assert_eq!(DGEMM_LDC.load(Ordering::SeqCst), 2);

    let alpha = Complex64::new(2.0, 3.0);
    let beta = Complex64::new(5.0, 7.0);
    let za = [Complex64::new(1.0, 0.0); 12];
    let zb = [Complex64::new(2.0, 0.0); 20];
    let mut zc = [Complex64::new(0.0, 0.0); 6];
    unsafe {
        cblas_zgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            2,
            3,
            4,
            &alpha,
            za.as_ptr(),
            4,
            zb.as_ptr(),
            3,
            &beta,
            zc.as_mut_ptr(),
            3,
        );
    }

    assert_eq!(zc[0], Complex64::new(16.0, 24.0));
    assert_eq!(ZGEMM_M.load(Ordering::SeqCst), 3);
    assert_eq!(ZGEMM_N.load(Ordering::SeqCst), 2);
    assert_eq!(ZGEMM_K.load(Ordering::SeqCst), 4);
    assert_eq!(ZGEMM_LDA.load(Ordering::SeqCst), 3);
    assert_eq!(ZGEMM_LDB.load(Ordering::SeqCst), 4);
    assert_eq!(ZGEMM_LDC.load(Ordering::SeqCst), 3);
    assert_eq!(f64::from_bits(ZGEMM_ALPHA_RE.load(Ordering::SeqCst)), 2.0);
    assert_eq!(f64::from_bits(ZGEMM_ALPHA_IM.load(Ordering::SeqCst)), 3.0);
    assert_eq!(f64::from_bits(ZGEMM_BETA_RE.load(Ordering::SeqCst)), 5.0);
    assert_eq!(f64::from_bits(ZGEMM_BETA_IM.load(Ordering::SeqCst)), 7.0);
    assert_eq!(ZGEMM_CALLS.load(Ordering::SeqCst), 1);

    let dgemm_calls = DGEMM_CALLS.load(Ordering::SeqCst);
    let mut overflow_c = [11.0; 1];
    unsafe {
        cblas_dgemm_64(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            i64::from(i32::MAX) + 1,
            1,
            1,
            1.0,
            a.as_ptr(),
            1,
            b.as_ptr(),
            1,
            0.0,
            overflow_c.as_mut_ptr(),
            1,
        );
    }
    assert_eq!(DGEMM_CALLS.load(Ordering::SeqCst), dgemm_calls);
    assert_eq!(overflow_c[0], 11.0);

    let zgemm_calls = ZGEMM_CALLS.load(Ordering::SeqCst);
    let mut overflow_zc = [Complex64::new(11.0, -11.0); 1];
    unsafe {
        cblas_zgemm_64(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            1,
            i64::from(i32::MAX) + 1,
            1,
            &alpha,
            za.as_ptr(),
            1,
            zb.as_ptr(),
            1,
            &beta,
            overflow_zc.as_mut_ptr(),
            1,
        );
    }
    assert_eq!(ZGEMM_CALLS.load(Ordering::SeqCst), zgemm_calls);
    assert_eq!(overflow_zc[0], Complex64::new(11.0, -11.0));

    let mut overflow_sc = [11.0_f32; 1];
    unsafe {
        cblas_sgemm_64(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            i64::from(i32::MAX) + 1,
            1,
            1,
            1.0,
            [1.0_f32].as_ptr(),
            1,
            [1.0_f32].as_ptr(),
            1,
            0.0,
            overflow_sc.as_mut_ptr(),
            1,
        );
    }
    assert_eq!(SGEMM_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(overflow_sc[0], 11.0);

    let alpha32 = Complex32::new(1.0, 2.0);
    let beta32 = Complex32::new(0.0, 0.0);
    let ca = [Complex32::new(1.0, 0.0); 1];
    let cb = [Complex32::new(1.0, 0.0); 1];
    let mut overflow_cc = [Complex32::new(11.0, -11.0); 1];
    unsafe {
        cblas_cgemm_64(
            CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            1,
            i64::from(i32::MAX) + 1,
            1,
            &alpha32,
            ca.as_ptr(),
            1,
            cb.as_ptr(),
            1,
            &beta32,
            overflow_cc.as_mut_ptr(),
            1,
        );
    }
    assert_eq!(CGEMM_CALLS.load(Ordering::SeqCst), 0);
    assert_eq!(overflow_cc[0], Complex32::new(11.0, -11.0));
}
