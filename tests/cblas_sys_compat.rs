#![cfg(all(not(feature = "ilp64"), not(feature = "openblas")))]

//! Test that cblas-inject's `#[no_mangle]` symbols satisfy cblas-sys's `extern "C"` declarations.
//!
//! This test does NOT link any external BLAS library. Instead, it registers a
//! hand-written Fortran dgemm_ with cblas-inject, then calls `cblas_sys::cblas_dgemm`
//! to verify the linker resolves it to cblas-inject's implementation.
//!
//! cblas-sys is LP64-only, and the `openblas` feature auto-registers providers at
//! load time, so this test only applies to the default manual-registration build.

// NOTE: We intentionally do NOT use `extern crate blas_src;` here.
// No native CBLAS library is linked — cblas-inject is the sole provider
// of the `cblas_dgemm` symbol that cblas-sys's extern declaration requires.

use std::ffi::c_char;

use cblas_inject::{blasint, register_dgemm};

/// Minimal Fortran-style dgemm implementation for testing (column-major, NoTrans only).
///
/// Computes: C = alpha * A * B + beta * C  (column-major storage)
unsafe extern "C" fn mock_dgemm(
    _transa: *const c_char,
    _transb: *const c_char,
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
) {
    let m = *m as usize;
    let n = *n as usize;
    let k = *k as usize;
    let alpha = *alpha;
    let beta = *beta;
    let lda = *lda as usize;
    let ldb = *ldb as usize;
    let ldc = *ldc as usize;

    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0;
            for p in 0..k {
                sum += *a.add(i + p * lda) * *b.add(p + j * ldb);
            }
            let c_ptr = c.add(i + j * ldc);
            *c_ptr = alpha * sum + beta * *c_ptr;
        }
    }
}

fn setup() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| unsafe {
        register_dgemm(mock_dgemm);
    });
}

/// Verify that `cblas_sys::cblas_dgemm` links to cblas-inject's implementation.
///
/// cblas-sys declares `extern "C" { fn cblas_dgemm(...); }` without providing
/// an implementation. cblas-inject's `#[no_mangle] pub extern "C" fn cblas_dgemm`
/// satisfies this declaration at link time.
#[test]
fn test_cblas_sys_dgemm_resolved_by_inject() {
    setup();

    // 2x2 row-major: C = A * B
    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [5.0, 6.0, 7.0, 8.0];
    let mut c = [0.0f64; 4];

    unsafe {
        cblas_sys::cblas_dgemm(
            cblas_sys::CblasRowMajor,
            cblas_sys::CblasNoTrans,
            cblas_sys::CblasNoTrans,
            2,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
    }

    // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assert_eq!(c, [19.0, 22.0, 43.0, 50.0]);
}

/// Also verify column-major works through cblas_sys.
#[test]
fn test_cblas_sys_dgemm_colmajor() {
    setup();

    // 2x2 column-major: C = A * B
    // A = [[1, 3], [2, 4]] stored as [1, 2, 3, 4] (col-major)
    // B = [[5, 7], [6, 8]] stored as [5, 6, 7, 8] (col-major)
    let a = [1.0, 2.0, 3.0, 4.0]; // col-major: A = [[1,3],[2,4]]
    let b = [5.0, 6.0, 7.0, 8.0]; // col-major: B = [[5,7],[6,8]]
    let mut c = [0.0f64; 4];

    unsafe {
        cblas_sys::cblas_dgemm(
            cblas_sys::CblasColMajor,
            cblas_sys::CblasNoTrans,
            cblas_sys::CblasNoTrans,
            2,
            2,
            2,
            1.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.0,
            c.as_mut_ptr(),
            2,
        );
    }

    // C = A*B = [[1*5+3*6, 1*7+3*8], [2*5+4*6, 2*7+4*8]] = [[23, 31], [34, 46]]
    // col-major: [23, 34, 31, 46]
    assert_eq!(c, [23.0, 34.0, 31.0, 46.0]);
}

/// Test with alpha and beta scaling through cblas_sys.
#[test]
fn test_cblas_sys_dgemm_alpha_beta() {
    setup();

    // C = 2.0 * A * B + 0.5 * C_init (row-major)
    let a = [1.0, 0.0, 0.0, 1.0]; // identity
    let b = [3.0, 4.0, 5.0, 6.0];
    let mut c = [10.0, 20.0, 30.0, 40.0];

    unsafe {
        cblas_sys::cblas_dgemm(
            cblas_sys::CblasRowMajor,
            cblas_sys::CblasNoTrans,
            cblas_sys::CblasNoTrans,
            2,
            2,
            2,
            2.0,
            a.as_ptr(),
            2,
            b.as_ptr(),
            2,
            0.5,
            c.as_mut_ptr(),
            2,
        );
    }

    // C = 2*I*B + 0.5*C_init = 2*B + 0.5*C_init
    // = [[6+5, 8+10], [10+15, 12+20]] = [[11, 18], [25, 32]]
    assert_eq!(c, [11.0, 18.0, 25.0, 32.0]);
}
