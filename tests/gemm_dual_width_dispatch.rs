#![cfg(not(feature = "openblas"))]

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};

#[cfg(feature = "ilp64")]
use cblas_inject::BlasInt32;
#[cfg(not(feature = "ilp64"))]
use cblas_inject::BlasInt64;
use cblas_inject::{
    cblas_dgemm, cblas_zgemm, CblasColMajor, CblasNoTrans, CblasRowMajor, CBLAS_INJECT_STATUS_OK,
};
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
unsafe extern "C" fn mock_dgemm_lp64(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    k: *const BlasInt32,
    _alpha: *const f64,
    _a: *const f64,
    lda: *const BlasInt32,
    _b: *const f64,
    ldb: *const BlasInt32,
    _beta: *const f64,
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
    unsafe {
        *c = 32.0;
    }
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn mock_dgemm_ilp64(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    k: *const BlasInt64,
    _alpha: *const f64,
    _a: *const f64,
    lda: *const BlasInt64,
    _b: *const f64,
    ldb: *const BlasInt64,
    _beta: *const f64,
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
    unsafe {
        *c = 64.0;
    }
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_zgemm_lp64(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const BlasInt32,
    n: *const BlasInt32,
    k: *const BlasInt32,
    alpha: *const Complex64,
    _a: *const Complex64,
    lda: *const BlasInt32,
    _b: *const Complex64,
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
    let alpha = unsafe { *alpha };
    let beta = unsafe { *beta };
    ZGEMM_ALPHA_RE.store(alpha.re.to_bits(), Ordering::SeqCst);
    ZGEMM_ALPHA_IM.store(alpha.im.to_bits(), Ordering::SeqCst);
    ZGEMM_BETA_RE.store(beta.re.to_bits(), Ordering::SeqCst);
    ZGEMM_BETA_IM.store(beta.im.to_bits(), Ordering::SeqCst);
    unsafe {
        *c = Complex64::new(32.0, -32.0);
    }
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn mock_zgemm_ilp64(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const BlasInt64,
    n: *const BlasInt64,
    k: *const BlasInt64,
    alpha: *const Complex64,
    _a: *const Complex64,
    lda: *const BlasInt64,
    _b: *const Complex64,
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
    let alpha = unsafe { *alpha };
    let beta = unsafe { *beta };
    ZGEMM_ALPHA_RE.store(alpha.re.to_bits(), Ordering::SeqCst);
    ZGEMM_ALPHA_IM.store(alpha.im.to_bits(), Ordering::SeqCst);
    ZGEMM_BETA_RE.store(beta.re.to_bits(), Ordering::SeqCst);
    ZGEMM_BETA_IM.store(beta.im.to_bits(), Ordering::SeqCst);
    unsafe {
        *c = Complex64::new(64.0, -64.0);
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

    assert_eq!(c[0], 64.0);
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

    assert_eq!(zc[0], Complex64::new(64.0, -64.0));
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

    assert_eq!(c[0], 32.0);
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

    assert_eq!(zc[0], Complex64::new(32.0, -32.0));
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
        cblas_dgemm(
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
        cblas_zgemm(
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
}
