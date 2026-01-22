//! Auto-registration of Fortran BLAS functions from OpenBLAS.
//!
//! This module uses the `ctor` crate to automatically register Fortran BLAS
//! function pointers when the library is loaded. This is required for the
//! cdylib build to work with OpenBLAS ctest.

use crate::backend::*;
use crate::types::ComplexReturnStyle;
use num_complex::{Complex32, Complex64};

// Fortran BLAS declarations (linked from OpenBLAS)
#[link(name = "openblas")]
extern "C" {
    // BLAS Level 1 - Single precision
    fn srot_(n: *const i32, x: *mut f32, incx: *const i32, y: *mut f32, incy: *const i32, c: *const f32, s: *const f32);
    fn srotg_(a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32);
    fn srotm_(n: *const i32, x: *mut f32, incx: *const i32, y: *mut f32, incy: *const i32, param: *const f32);
    fn srotmg_(d1: *mut f32, d2: *mut f32, x1: *mut f32, y1: *const f32, param: *mut f32);
    fn sswap_(n: *const i32, x: *mut f32, incx: *const i32, y: *mut f32, incy: *const i32);
    fn scopy_(n: *const i32, x: *const f32, incx: *const i32, y: *mut f32, incy: *const i32);
    fn saxpy_(n: *const i32, alpha: *const f32, x: *const f32, incx: *const i32, y: *mut f32, incy: *const i32);
    fn sscal_(n: *const i32, alpha: *const f32, x: *mut f32, incx: *const i32);
    fn sdot_(n: *const i32, x: *const f32, incx: *const i32, y: *const f32, incy: *const i32) -> f32;
    fn sdsdot_(n: *const i32, sb: *const f32, x: *const f32, incx: *const i32, y: *const f32, incy: *const i32) -> f32;
    fn snrm2_(n: *const i32, x: *const f32, incx: *const i32) -> f32;
    fn sasum_(n: *const i32, x: *const f32, incx: *const i32) -> f32;
    fn isamax_(n: *const i32, x: *const f32, incx: *const i32) -> i32;

    // BLAS Level 1 - Double precision
    fn drot_(n: *const i32, x: *mut f64, incx: *const i32, y: *mut f64, incy: *const i32, c: *const f64, s: *const f64);
    fn drotg_(a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64);
    fn drotm_(n: *const i32, x: *mut f64, incx: *const i32, y: *mut f64, incy: *const i32, param: *const f64);
    fn drotmg_(d1: *mut f64, d2: *mut f64, x1: *mut f64, y1: *const f64, param: *mut f64);
    fn dswap_(n: *const i32, x: *mut f64, incx: *const i32, y: *mut f64, incy: *const i32);
    fn dcopy_(n: *const i32, x: *const f64, incx: *const i32, y: *mut f64, incy: *const i32);
    fn daxpy_(n: *const i32, alpha: *const f64, x: *const f64, incx: *const i32, y: *mut f64, incy: *const i32);
    fn dscal_(n: *const i32, alpha: *const f64, x: *mut f64, incx: *const i32);
    fn ddot_(n: *const i32, x: *const f64, incx: *const i32, y: *const f64, incy: *const i32) -> f64;
    fn dsdot_(n: *const i32, x: *const f32, incx: *const i32, y: *const f32, incy: *const i32) -> f64;
    fn dnrm2_(n: *const i32, x: *const f64, incx: *const i32) -> f64;
    fn dasum_(n: *const i32, x: *const f64, incx: *const i32) -> f64;
    fn idamax_(n: *const i32, x: *const f64, incx: *const i32) -> i32;

    // BLAS Level 1 - Single complex
    fn cswap_(n: *const i32, x: *mut (), incx: *const i32, y: *mut (), incy: *const i32);
    fn ccopy_(n: *const i32, x: *const (), incx: *const i32, y: *mut (), incy: *const i32);
    fn caxpy_(n: *const i32, alpha: *const (), x: *const (), incx: *const i32, y: *mut (), incy: *const i32);
    fn cscal_(n: *const i32, alpha: *const (), x: *mut (), incx: *const i32);
    fn csscal_(n: *const i32, alpha: *const f32, x: *mut (), incx: *const i32);
    // Note: On ARM64/x86_64 with gfortran, complex dot products return value by register
    fn cdotu_(n: *const i32, x: *const (), incx: *const i32, y: *const (), incy: *const i32) -> Complex32;
    fn cdotc_(n: *const i32, x: *const (), incx: *const i32, y: *const (), incy: *const i32) -> Complex32;
    fn scnrm2_(n: *const i32, x: *const (), incx: *const i32) -> f32;
    fn scasum_(n: *const i32, x: *const (), incx: *const i32) -> f32;
    fn icamax_(n: *const i32, x: *const (), incx: *const i32) -> i32;

    // BLAS Level 1 - Double complex
    fn zswap_(n: *const i32, x: *mut (), incx: *const i32, y: *mut (), incy: *const i32);
    fn zcopy_(n: *const i32, x: *const (), incx: *const i32, y: *mut (), incy: *const i32);
    fn zaxpy_(n: *const i32, alpha: *const (), x: *const (), incx: *const i32, y: *mut (), incy: *const i32);
    fn zscal_(n: *const i32, alpha: *const (), x: *mut (), incx: *const i32);
    fn zdscal_(n: *const i32, alpha: *const f64, x: *mut (), incx: *const i32);
    // Note: On ARM64/x86_64 with gfortran, complex dot products return value by register
    fn zdotu_(n: *const i32, x: *const (), incx: *const i32, y: *const (), incy: *const i32) -> Complex64;
    fn zdotc_(n: *const i32, x: *const (), incx: *const i32, y: *const (), incy: *const i32) -> Complex64;
    fn dznrm2_(n: *const i32, x: *const (), incx: *const i32) -> f64;
    fn dzasum_(n: *const i32, x: *const (), incx: *const i32) -> f64;
    fn izamax_(n: *const i32, x: *const (), incx: *const i32) -> i32;

    // BLAS Level 3
    fn sgemm_(transa: *const i8, transb: *const i8, m: *const i32, n: *const i32, k: *const i32,
              alpha: *const f32, a: *const f32, lda: *const i32, b: *const f32, ldb: *const i32,
              beta: *const f32, c: *mut f32, ldc: *const i32);
    fn dgemm_(transa: *const i8, transb: *const i8, m: *const i32, n: *const i32, k: *const i32,
              alpha: *const f64, a: *const f64, lda: *const i32, b: *const f64, ldb: *const i32,
              beta: *const f64, c: *mut f64, ldc: *const i32);
    fn cgemm_(transa: *const i8, transb: *const i8, m: *const i32, n: *const i32, k: *const i32,
              alpha: *const (), a: *const (), lda: *const i32, b: *const (), ldb: *const i32,
              beta: *const (), c: *mut (), ldc: *const i32);
    fn zgemm_(transa: *const i8, transb: *const i8, m: *const i32, n: *const i32, k: *const i32,
              alpha: *const (), a: *const (), lda: *const i32, b: *const (), ldb: *const i32,
              beta: *const (), c: *mut (), ldc: *const i32);

    fn dsymm_(side: *const i8, uplo: *const i8, m: *const i32, n: *const i32,
              alpha: *const f64, a: *const f64, lda: *const i32, b: *const f64, ldb: *const i32,
              beta: *const f64, c: *mut f64, ldc: *const i32);
    fn dsyrk_(uplo: *const i8, trans: *const i8, n: *const i32, k: *const i32,
              alpha: *const f64, a: *const f64, lda: *const i32,
              beta: *const f64, c: *mut f64, ldc: *const i32);
    fn dsyr2k_(uplo: *const i8, trans: *const i8, n: *const i32, k: *const i32,
               alpha: *const f64, a: *const f64, lda: *const i32, b: *const f64, ldb: *const i32,
               beta: *const f64, c: *mut f64, ldc: *const i32);
    fn dtrmm_(side: *const i8, uplo: *const i8, transa: *const i8, diag: *const i8,
              m: *const i32, n: *const i32, alpha: *const f64, a: *const f64, lda: *const i32,
              b: *mut f64, ldb: *const i32);
    fn dtrsm_(side: *const i8, uplo: *const i8, transa: *const i8, diag: *const i8,
              m: *const i32, n: *const i32, alpha: *const f64, a: *const f64, lda: *const i32,
              b: *mut f64, ldb: *const i32);
}

#[ctor::ctor]
fn register_all_blas() {
    // OpenBLAS uses return value convention for complex dot products
    set_complex_return_style(ComplexReturnStyle::ReturnValue);

    unsafe {
        // BLAS Level 1 - Single
        register_srot(std::mem::transmute(srot_ as *const ()));
        register_srotg(std::mem::transmute(srotg_ as *const ()));
        register_srotm(std::mem::transmute(srotm_ as *const ()));
        register_srotmg(std::mem::transmute(srotmg_ as *const ()));
        register_sswap(std::mem::transmute(sswap_ as *const ()));
        register_scopy(std::mem::transmute(scopy_ as *const ()));
        register_saxpy(std::mem::transmute(saxpy_ as *const ()));
        register_sscal(std::mem::transmute(sscal_ as *const ()));
        register_sdot(std::mem::transmute(sdot_ as *const ()));
        register_sdsdot(std::mem::transmute(sdsdot_ as *const ()));
        register_snrm2(std::mem::transmute(snrm2_ as *const ()));
        register_sasum(std::mem::transmute(sasum_ as *const ()));
        register_isamax(std::mem::transmute(isamax_ as *const ()));

        // BLAS Level 1 - Double
        register_drot(std::mem::transmute(drot_ as *const ()));
        register_drotg(std::mem::transmute(drotg_ as *const ()));
        register_drotm(std::mem::transmute(drotm_ as *const ()));
        register_drotmg(std::mem::transmute(drotmg_ as *const ()));
        register_dswap(std::mem::transmute(dswap_ as *const ()));
        register_dcopy(std::mem::transmute(dcopy_ as *const ()));
        register_daxpy(std::mem::transmute(daxpy_ as *const ()));
        register_dscal(std::mem::transmute(dscal_ as *const ()));
        register_ddot(std::mem::transmute(ddot_ as *const ()));
        register_dsdot(std::mem::transmute(dsdot_ as *const ()));
        register_dnrm2(std::mem::transmute(dnrm2_ as *const ()));
        register_dasum(std::mem::transmute(dasum_ as *const ()));
        register_idamax(std::mem::transmute(idamax_ as *const ()));

        // BLAS Level 1 - Single complex
        register_cswap(std::mem::transmute(cswap_ as *const ()));
        register_ccopy(std::mem::transmute(ccopy_ as *const ()));
        register_caxpy(std::mem::transmute(caxpy_ as *const ()));
        register_cscal(std::mem::transmute(cscal_ as *const ()));
        register_csscal(std::mem::transmute(csscal_ as *const ()));
        register_cdotu(std::mem::transmute(cdotu_ as *const ()));
        register_cdotc(std::mem::transmute(cdotc_ as *const ()));
        register_scnrm2(std::mem::transmute(scnrm2_ as *const ()));
        register_scasum(std::mem::transmute(scasum_ as *const ()));
        register_icamax(std::mem::transmute(icamax_ as *const ()));

        // BLAS Level 1 - Double complex
        register_zswap(std::mem::transmute(zswap_ as *const ()));
        register_zcopy(std::mem::transmute(zcopy_ as *const ()));
        register_zaxpy(std::mem::transmute(zaxpy_ as *const ()));
        register_zscal(std::mem::transmute(zscal_ as *const ()));
        register_zdscal(std::mem::transmute(zdscal_ as *const ()));
        register_zdotu(std::mem::transmute(zdotu_ as *const ()));
        register_zdotc(std::mem::transmute(zdotc_ as *const ()));
        register_dznrm2(std::mem::transmute(dznrm2_ as *const ()));
        register_dzasum(std::mem::transmute(dzasum_ as *const ()));
        register_izamax(std::mem::transmute(izamax_ as *const ()));

        // BLAS Level 3
        register_sgemm(std::mem::transmute(sgemm_ as *const ()));
        register_dgemm(std::mem::transmute(dgemm_ as *const ()));
        register_cgemm(std::mem::transmute(cgemm_ as *const ()));
        register_zgemm(std::mem::transmute(zgemm_ as *const ()));
        register_dsymm(std::mem::transmute(dsymm_ as *const ()));
        register_dsyrk(std::mem::transmute(dsyrk_ as *const ()));
        register_dsyr2k(std::mem::transmute(dsyr2k_ as *const ()));
        register_dtrmm(std::mem::transmute(dtrmm_ as *const ()));
        register_dtrsm(std::mem::transmute(dtrsm_ as *const ()));
    }
}
