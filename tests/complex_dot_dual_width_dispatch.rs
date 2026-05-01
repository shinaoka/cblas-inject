#![cfg(not(feature = "openblas"))]

use std::ffi::c_void;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

use cblas_inject::{
    cblas_cdotc_sub, cblas_cdotc_sub_64, cblas_cdotu_sub, cblas_cdotu_sub_64, cblas_zdotc_sub,
    cblas_zdotc_sub_64, cblas_zdotu_sub, cblas_zdotu_sub_64, BlasInt32, BlasInt64,
    CBLAS_INJECT_STATUS_OK,
};
use num_complex::{Complex32, Complex64};

static CDOTU_CALLS: AtomicUsize = AtomicUsize::new(0);
static CDOTU_N: AtomicI64 = AtomicI64::new(0);
static CDOTU_INCX: AtomicI64 = AtomicI64::new(0);
static CDOTU_INCY: AtomicI64 = AtomicI64::new(0);

static CDOTC_CALLS: AtomicUsize = AtomicUsize::new(0);
static CDOTC_N: AtomicI64 = AtomicI64::new(0);
static CDOTC_INCX: AtomicI64 = AtomicI64::new(0);
static CDOTC_INCY: AtomicI64 = AtomicI64::new(0);

static ZDOTU_CALLS: AtomicUsize = AtomicUsize::new(0);
static ZDOTU_N: AtomicI64 = AtomicI64::new(0);
static ZDOTU_INCX: AtomicI64 = AtomicI64::new(0);
static ZDOTU_INCY: AtomicI64 = AtomicI64::new(0);

static ZDOTC_CALLS: AtomicUsize = AtomicUsize::new(0);
static ZDOTC_N: AtomicI64 = AtomicI64::new(0);
static ZDOTC_INCX: AtomicI64 = AtomicI64::new(0);
static ZDOTC_INCY: AtomicI64 = AtomicI64::new(0);

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn mock_cdotu_ilp64(
    n: *const BlasInt64,
    _x: *const Complex32,
    incx: *const BlasInt64,
    _y: *const Complex32,
    incy: *const BlasInt64,
) -> Complex32 {
    CDOTU_CALLS.fetch_add(1, Ordering::SeqCst);
    CDOTU_N.store(unsafe { *n }, Ordering::SeqCst);
    CDOTU_INCX.store(unsafe { *incx }, Ordering::SeqCst);
    CDOTU_INCY.store(unsafe { *incy }, Ordering::SeqCst);
    Complex32::new(64.0, -64.0)
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_cdotu_lp64(
    n: *const BlasInt32,
    _x: *const Complex32,
    incx: *const BlasInt32,
    _y: *const Complex32,
    incy: *const BlasInt32,
) -> Complex32 {
    CDOTU_CALLS.fetch_add(1, Ordering::SeqCst);
    CDOTU_N.store(i64::from(unsafe { *n }), Ordering::SeqCst);
    CDOTU_INCX.store(i64::from(unsafe { *incx }), Ordering::SeqCst);
    CDOTU_INCY.store(i64::from(unsafe { *incy }), Ordering::SeqCst);
    Complex32::new(32.0, -32.0)
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn mock_cdotc_ilp64(
    n: *const BlasInt64,
    _x: *const Complex32,
    incx: *const BlasInt64,
    _y: *const Complex32,
    incy: *const BlasInt64,
) -> Complex32 {
    CDOTC_CALLS.fetch_add(1, Ordering::SeqCst);
    CDOTC_N.store(unsafe { *n }, Ordering::SeqCst);
    CDOTC_INCX.store(unsafe { *incx }, Ordering::SeqCst);
    CDOTC_INCY.store(unsafe { *incy }, Ordering::SeqCst);
    Complex32::new(65.0, -65.0)
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_cdotc_lp64(
    n: *const BlasInt32,
    _x: *const Complex32,
    incx: *const BlasInt32,
    _y: *const Complex32,
    incy: *const BlasInt32,
) -> Complex32 {
    CDOTC_CALLS.fetch_add(1, Ordering::SeqCst);
    CDOTC_N.store(i64::from(unsafe { *n }), Ordering::SeqCst);
    CDOTC_INCX.store(i64::from(unsafe { *incx }), Ordering::SeqCst);
    CDOTC_INCY.store(i64::from(unsafe { *incy }), Ordering::SeqCst);
    Complex32::new(33.0, -33.0)
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn mock_zdotu_ilp64(
    n: *const BlasInt64,
    _x: *const Complex64,
    incx: *const BlasInt64,
    _y: *const Complex64,
    incy: *const BlasInt64,
) -> Complex64 {
    ZDOTU_CALLS.fetch_add(1, Ordering::SeqCst);
    ZDOTU_N.store(unsafe { *n }, Ordering::SeqCst);
    ZDOTU_INCX.store(unsafe { *incx }, Ordering::SeqCst);
    ZDOTU_INCY.store(unsafe { *incy }, Ordering::SeqCst);
    Complex64::new(66.0, -66.0)
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_zdotu_lp64(
    n: *const BlasInt32,
    _x: *const Complex64,
    incx: *const BlasInt32,
    _y: *const Complex64,
    incy: *const BlasInt32,
) -> Complex64 {
    ZDOTU_CALLS.fetch_add(1, Ordering::SeqCst);
    ZDOTU_N.store(i64::from(unsafe { *n }), Ordering::SeqCst);
    ZDOTU_INCX.store(i64::from(unsafe { *incx }), Ordering::SeqCst);
    ZDOTU_INCY.store(i64::from(unsafe { *incy }), Ordering::SeqCst);
    Complex64::new(34.0, -34.0)
}

#[cfg(not(feature = "ilp64"))]
unsafe extern "C" fn mock_zdotc_ilp64(
    n: *const BlasInt64,
    _x: *const Complex64,
    incx: *const BlasInt64,
    _y: *const Complex64,
    incy: *const BlasInt64,
) -> Complex64 {
    ZDOTC_CALLS.fetch_add(1, Ordering::SeqCst);
    ZDOTC_N.store(unsafe { *n }, Ordering::SeqCst);
    ZDOTC_INCX.store(unsafe { *incx }, Ordering::SeqCst);
    ZDOTC_INCY.store(unsafe { *incy }, Ordering::SeqCst);
    Complex64::new(67.0, -67.0)
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_zdotc_lp64(
    n: *const BlasInt32,
    _x: *const Complex64,
    incx: *const BlasInt32,
    _y: *const Complex64,
    incy: *const BlasInt32,
) -> Complex64 {
    ZDOTC_CALLS.fetch_add(1, Ordering::SeqCst);
    ZDOTC_N.store(i64::from(unsafe { *n }), Ordering::SeqCst);
    ZDOTC_INCX.store(i64::from(unsafe { *incx }), Ordering::SeqCst);
    ZDOTC_INCY.store(i64::from(unsafe { *incy }), Ordering::SeqCst);
    Complex64::new(35.0, -35.0)
}

#[cfg(not(feature = "ilp64"))]
#[test]
fn lp64_complex_dot_wrappers_widen_to_ilp64_fallback_providers() {
    unsafe {
        assert_eq!(
            cblas_inject::cblas_inject_register_cdotu_ilp64(mock_cdotu_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_cdotc_ilp64(mock_cdotc_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zdotu_ilp64(mock_zdotu_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zdotc_ilp64(mock_zdotc_ilp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
    }

    let x32 = [Complex32::new(1.0, 2.0); 8];
    let y32 = [Complex32::new(3.0, 4.0); 8];
    let x64 = [Complex64::new(1.0, 2.0); 8];
    let y64 = [Complex64::new(3.0, 4.0); 8];

    let mut cdotu = Complex32::new(0.0, 0.0);
    unsafe {
        cblas_cdotu_sub(5, x32.as_ptr(), 2, y32.as_ptr(), 3, &mut cdotu);
    }
    assert_eq!(cdotu, Complex32::new(64.0, -64.0));
    assert_eq!(CDOTU_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CDOTU_N.load(Ordering::SeqCst), 5);
    assert_eq!(CDOTU_INCX.load(Ordering::SeqCst), 2);
    assert_eq!(CDOTU_INCY.load(Ordering::SeqCst), 3);

    let mut cdotc = Complex32::new(0.0, 0.0);
    unsafe {
        cblas_cdotc_sub(6, x32.as_ptr(), 3, y32.as_ptr(), 4, &mut cdotc);
    }
    assert_eq!(cdotc, Complex32::new(65.0, -65.0));
    assert_eq!(CDOTC_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CDOTC_N.load(Ordering::SeqCst), 6);
    assert_eq!(CDOTC_INCX.load(Ordering::SeqCst), 3);
    assert_eq!(CDOTC_INCY.load(Ordering::SeqCst), 4);

    let mut zdotu = Complex64::new(0.0, 0.0);
    unsafe {
        cblas_zdotu_sub(7, x64.as_ptr(), 4, y64.as_ptr(), 5, &mut zdotu);
    }
    assert_eq!(zdotu, Complex64::new(66.0, -66.0));
    assert_eq!(ZDOTU_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(ZDOTU_N.load(Ordering::SeqCst), 7);
    assert_eq!(ZDOTU_INCX.load(Ordering::SeqCst), 4);
    assert_eq!(ZDOTU_INCY.load(Ordering::SeqCst), 5);

    let mut zdotc = Complex64::new(0.0, 0.0);
    unsafe {
        cblas_zdotc_sub(8, x64.as_ptr(), 5, y64.as_ptr(), 6, &mut zdotc);
    }
    assert_eq!(zdotc, Complex64::new(67.0, -67.0));
    assert_eq!(ZDOTC_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(ZDOTC_N.load(Ordering::SeqCst), 8);
    assert_eq!(ZDOTC_INCX.load(Ordering::SeqCst), 5);
    assert_eq!(ZDOTC_INCY.load(Ordering::SeqCst), 6);
}

#[cfg(feature = "ilp64")]
#[test]
fn ilp64_complex_dot_wrappers_narrow_to_lp64_fallback_providers() {
    unsafe {
        assert_eq!(
            cblas_inject::cblas_inject_register_cdotu_lp64(mock_cdotu_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_cdotc_lp64(mock_cdotc_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zdotu_lp64(mock_zdotu_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            cblas_inject::cblas_inject_register_zdotc_lp64(mock_zdotc_lp64 as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
    }

    let x32 = [Complex32::new(1.0, 2.0); 8];
    let y32 = [Complex32::new(3.0, 4.0); 8];
    let x64 = [Complex64::new(1.0, 2.0); 8];
    let y64 = [Complex64::new(3.0, 4.0); 8];

    let mut cdotu = Complex32::new(0.0, 0.0);
    unsafe {
        cblas_cdotu_sub_64(5, x32.as_ptr(), 2, y32.as_ptr(), 3, &mut cdotu);
    }
    assert_eq!(cdotu, Complex32::new(32.0, -32.0));
    assert_eq!(CDOTU_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CDOTU_N.load(Ordering::SeqCst), 5);
    assert_eq!(CDOTU_INCX.load(Ordering::SeqCst), 2);
    assert_eq!(CDOTU_INCY.load(Ordering::SeqCst), 3);

    let mut cdotc = Complex32::new(0.0, 0.0);
    unsafe {
        cblas_cdotc_sub_64(6, x32.as_ptr(), 3, y32.as_ptr(), 4, &mut cdotc);
    }
    assert_eq!(cdotc, Complex32::new(33.0, -33.0));
    assert_eq!(CDOTC_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(CDOTC_N.load(Ordering::SeqCst), 6);
    assert_eq!(CDOTC_INCX.load(Ordering::SeqCst), 3);
    assert_eq!(CDOTC_INCY.load(Ordering::SeqCst), 4);

    let mut zdotu = Complex64::new(0.0, 0.0);
    unsafe {
        cblas_zdotu_sub_64(7, x64.as_ptr(), 4, y64.as_ptr(), 5, &mut zdotu);
    }
    assert_eq!(zdotu, Complex64::new(34.0, -34.0));
    assert_eq!(ZDOTU_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(ZDOTU_N.load(Ordering::SeqCst), 7);
    assert_eq!(ZDOTU_INCX.load(Ordering::SeqCst), 4);
    assert_eq!(ZDOTU_INCY.load(Ordering::SeqCst), 5);

    let mut zdotc = Complex64::new(0.0, 0.0);
    unsafe {
        cblas_zdotc_sub_64(8, x64.as_ptr(), 5, y64.as_ptr(), 6, &mut zdotc);
    }
    assert_eq!(zdotc, Complex64::new(35.0, -35.0));
    assert_eq!(ZDOTC_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(ZDOTC_N.load(Ordering::SeqCst), 8);
    assert_eq!(ZDOTC_INCX.load(Ordering::SeqCst), 5);
    assert_eq!(ZDOTC_INCY.load(Ordering::SeqCst), 6);

    let mut overflow_c = Complex32::new(11.0, -11.0);
    unsafe {
        cblas_cdotu_sub_64(
            i64::from(i32::MAX) + 1,
            x32.as_ptr(),
            1,
            y32.as_ptr(),
            1,
            &mut overflow_c,
        );
    }
    assert_eq!(CDOTU_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(overflow_c, Complex32::new(11.0, -11.0));

    let mut overflow_z = Complex64::new(12.0, -12.0);
    unsafe {
        cblas_zdotc_sub_64(
            i64::from(i32::MAX) + 1,
            x64.as_ptr(),
            1,
            y64.as_ptr(),
            1,
            &mut overflow_z,
        );
    }
    assert_eq!(ZDOTC_CALLS.load(Ordering::SeqCst), 1);
    assert_eq!(overflow_z, Complex64::new(12.0, -12.0));
}
