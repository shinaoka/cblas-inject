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
    fn srot_(
        n: *const i32,
        x: *mut f32,
        incx: *const i32,
        y: *mut f32,
        incy: *const i32,
        c: *const f32,
        s: *const f32,
    );
    fn srotg_(a: *mut f32, b: *mut f32, c: *mut f32, s: *mut f32);
    fn srotm_(
        n: *const i32,
        x: *mut f32,
        incx: *const i32,
        y: *mut f32,
        incy: *const i32,
        param: *const f32,
    );
    fn srotmg_(d1: *mut f32, d2: *mut f32, x1: *mut f32, y1: *const f32, param: *mut f32);
    fn sswap_(n: *const i32, x: *mut f32, incx: *const i32, y: *mut f32, incy: *const i32);
    fn scopy_(n: *const i32, x: *const f32, incx: *const i32, y: *mut f32, incy: *const i32);
    fn saxpy_(
        n: *const i32,
        alpha: *const f32,
        x: *const f32,
        incx: *const i32,
        y: *mut f32,
        incy: *const i32,
    );
    fn sscal_(n: *const i32, alpha: *const f32, x: *mut f32, incx: *const i32);
    fn sdot_(
        n: *const i32,
        x: *const f32,
        incx: *const i32,
        y: *const f32,
        incy: *const i32,
    ) -> f32;
    fn sdsdot_(
        n: *const i32,
        sb: *const f32,
        x: *const f32,
        incx: *const i32,
        y: *const f32,
        incy: *const i32,
    ) -> f32;
    fn snrm2_(n: *const i32, x: *const f32, incx: *const i32) -> f32;
    fn sasum_(n: *const i32, x: *const f32, incx: *const i32) -> f32;
    fn isamax_(n: *const i32, x: *const f32, incx: *const i32) -> i32;

    // BLAS Level 1 - Double precision
    fn drot_(
        n: *const i32,
        x: *mut f64,
        incx: *const i32,
        y: *mut f64,
        incy: *const i32,
        c: *const f64,
        s: *const f64,
    );
    fn drotg_(a: *mut f64, b: *mut f64, c: *mut f64, s: *mut f64);
    fn drotm_(
        n: *const i32,
        x: *mut f64,
        incx: *const i32,
        y: *mut f64,
        incy: *const i32,
        param: *const f64,
    );
    fn drotmg_(d1: *mut f64, d2: *mut f64, x1: *mut f64, y1: *const f64, param: *mut f64);
    fn dswap_(n: *const i32, x: *mut f64, incx: *const i32, y: *mut f64, incy: *const i32);
    fn dcopy_(n: *const i32, x: *const f64, incx: *const i32, y: *mut f64, incy: *const i32);
    fn daxpy_(
        n: *const i32,
        alpha: *const f64,
        x: *const f64,
        incx: *const i32,
        y: *mut f64,
        incy: *const i32,
    );
    fn dscal_(n: *const i32, alpha: *const f64, x: *mut f64, incx: *const i32);
    fn ddot_(
        n: *const i32,
        x: *const f64,
        incx: *const i32,
        y: *const f64,
        incy: *const i32,
    ) -> f64;
    fn dsdot_(
        n: *const i32,
        x: *const f32,
        incx: *const i32,
        y: *const f32,
        incy: *const i32,
    ) -> f64;
    fn dnrm2_(n: *const i32, x: *const f64, incx: *const i32) -> f64;
    fn dasum_(n: *const i32, x: *const f64, incx: *const i32) -> f64;
    fn idamax_(n: *const i32, x: *const f64, incx: *const i32) -> i32;

    // BLAS Level 1 - Single complex
    fn cswap_(n: *const i32, x: *mut (), incx: *const i32, y: *mut (), incy: *const i32);
    fn ccopy_(n: *const i32, x: *const (), incx: *const i32, y: *mut (), incy: *const i32);
    fn caxpy_(
        n: *const i32,
        alpha: *const (),
        x: *const (),
        incx: *const i32,
        y: *mut (),
        incy: *const i32,
    );
    fn cscal_(n: *const i32, alpha: *const (), x: *mut (), incx: *const i32);
    fn csscal_(n: *const i32, alpha: *const f32, x: *mut (), incx: *const i32);
    // Note: On ARM64/x86_64 with gfortran, complex dot products return value by register
    fn cdotu_(
        n: *const i32,
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
    ) -> Complex32;
    fn cdotc_(
        n: *const i32,
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
    ) -> Complex32;
    fn scnrm2_(n: *const i32, x: *const (), incx: *const i32) -> f32;
    fn scasum_(n: *const i32, x: *const (), incx: *const i32) -> f32;
    fn icamax_(n: *const i32, x: *const (), incx: *const i32) -> i32;

    // BLAS Level 1 - Double complex
    fn zswap_(n: *const i32, x: *mut (), incx: *const i32, y: *mut (), incy: *const i32);
    fn zcopy_(n: *const i32, x: *const (), incx: *const i32, y: *mut (), incy: *const i32);
    fn zaxpy_(
        n: *const i32,
        alpha: *const (),
        x: *const (),
        incx: *const i32,
        y: *mut (),
        incy: *const i32,
    );
    fn zscal_(n: *const i32, alpha: *const (), x: *mut (), incx: *const i32);
    fn zdscal_(n: *const i32, alpha: *const f64, x: *mut (), incx: *const i32);
    // Note: On ARM64/x86_64 with gfortran, complex dot products return value by register
    fn zdotu_(
        n: *const i32,
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
    ) -> Complex64;
    fn zdotc_(
        n: *const i32,
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
    ) -> Complex64;
    fn dznrm2_(n: *const i32, x: *const (), incx: *const i32) -> f64;
    fn dzasum_(n: *const i32, x: *const (), incx: *const i32) -> f64;
    fn izamax_(n: *const i32, x: *const (), incx: *const i32) -> i32;

    // BLAS Level 3
    fn sgemm_(
        transa: *const i8,
        transb: *const i8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        b: *const f32,
        ldb: *const i32,
        beta: *const f32,
        c: *mut f32,
        ldc: *const i32,
    );
    fn dgemm_(
        transa: *const i8,
        transb: *const i8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn cgemm_(
        transa: *const i8,
        transb: *const i8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const (),
        a: *const (),
        lda: *const i32,
        b: *const (),
        ldb: *const i32,
        beta: *const (),
        c: *mut (),
        ldc: *const i32,
    );
    fn zgemm_(
        transa: *const i8,
        transb: *const i8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const (),
        a: *const (),
        lda: *const i32,
        b: *const (),
        ldb: *const i32,
        beta: *const (),
        c: *mut (),
        ldc: *const i32,
    );

    fn dsymm_(
        side: *const i8,
        uplo: *const i8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dsyrk_(
        uplo: *const i8,
        trans: *const i8,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dsyr2k_(
        uplo: *const i8,
        trans: *const i8,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *const f64,
        ldb: *const i32,
        beta: *const f64,
        c: *mut f64,
        ldc: *const i32,
    );
    fn dtrmm_(
        side: *const i8,
        uplo: *const i8,
        transa: *const i8,
        diag: *const i8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
    );
    fn dtrsm_(
        side: *const i8,
        uplo: *const i8,
        transa: *const i8,
        diag: *const i8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        b: *mut f64,
        ldb: *const i32,
    );

    // BLAS Level 2 - General matrix-vector multiply
    fn sgemv_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        x: *const f32,
        incx: *const i32,
        beta: *const f32,
        y: *mut f32,
        incy: *const i32,
    );
    fn dgemv_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        x: *const f64,
        incx: *const i32,
        beta: *const f64,
        y: *mut f64,
        incy: *const i32,
    );
    fn cgemv_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        alpha: *const (),
        a: *const (),
        lda: *const i32,
        x: *const (),
        incx: *const i32,
        beta: *const (),
        y: *mut (),
        incy: *const i32,
    );
    fn zgemv_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        alpha: *const (),
        a: *const (),
        lda: *const i32,
        x: *const (),
        incx: *const i32,
        beta: *const (),
        y: *mut (),
        incy: *const i32,
    );

    // BLAS Level 2 - General band matrix-vector multiply
    fn sgbmv_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        kl: *const i32,
        ku: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        x: *const f32,
        incx: *const i32,
        beta: *const f32,
        y: *mut f32,
        incy: *const i32,
    );
    fn dgbmv_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        kl: *const i32,
        ku: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        x: *const f64,
        incx: *const i32,
        beta: *const f64,
        y: *mut f64,
        incy: *const i32,
    );
    fn cgbmv_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        kl: *const i32,
        ku: *const i32,
        alpha: *const (),
        a: *const (),
        lda: *const i32,
        x: *const (),
        incx: *const i32,
        beta: *const (),
        y: *mut (),
        incy: *const i32,
    );
    fn zgbmv_(
        trans: *const i8,
        m: *const i32,
        n: *const i32,
        kl: *const i32,
        ku: *const i32,
        alpha: *const (),
        a: *const (),
        lda: *const i32,
        x: *const (),
        incx: *const i32,
        beta: *const (),
        y: *mut (),
        incy: *const i32,
    );

    // BLAS Level 2 - Symmetric/Hermitian matrix-vector
    fn ssymv_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        x: *const f32,
        incx: *const i32,
        beta: *const f32,
        y: *mut f32,
        incy: *const i32,
    );
    fn dsymv_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        x: *const f64,
        incx: *const i32,
        beta: *const f64,
        y: *mut f64,
        incy: *const i32,
    );
    fn chemv_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const (),
        a: *const (),
        lda: *const i32,
        x: *const (),
        incx: *const i32,
        beta: *const (),
        y: *mut (),
        incy: *const i32,
    );
    fn zhemv_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const (),
        a: *const (),
        lda: *const i32,
        x: *const (),
        incx: *const i32,
        beta: *const (),
        y: *mut (),
        incy: *const i32,
    );

    // BLAS Level 2 - Symmetric/Hermitian band matrix-vector
    fn ssbmv_(
        uplo: *const i8,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        x: *const f32,
        incx: *const i32,
        beta: *const f32,
        y: *mut f32,
        incy: *const i32,
    );
    fn dsbmv_(
        uplo: *const i8,
        n: *const i32,
        k: *const i32,
        alpha: *const f64,
        a: *const f64,
        lda: *const i32,
        x: *const f64,
        incx: *const i32,
        beta: *const f64,
        y: *mut f64,
        incy: *const i32,
    );
    fn chbmv_(
        uplo: *const i8,
        n: *const i32,
        k: *const i32,
        alpha: *const (),
        a: *const (),
        lda: *const i32,
        x: *const (),
        incx: *const i32,
        beta: *const (),
        y: *mut (),
        incy: *const i32,
    );
    fn zhbmv_(
        uplo: *const i8,
        n: *const i32,
        k: *const i32,
        alpha: *const (),
        a: *const (),
        lda: *const i32,
        x: *const (),
        incx: *const i32,
        beta: *const (),
        y: *mut (),
        incy: *const i32,
    );

    // BLAS Level 2 - Triangular matrix-vector
    fn strmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        a: *const f32,
        lda: *const i32,
        x: *mut f32,
        incx: *const i32,
    );
    fn dtrmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        a: *const f64,
        lda: *const i32,
        x: *mut f64,
        incx: *const i32,
    );
    fn ctrmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        a: *const (),
        lda: *const i32,
        x: *mut (),
        incx: *const i32,
    );
    fn ztrmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        a: *const (),
        lda: *const i32,
        x: *mut (),
        incx: *const i32,
    );

    // BLAS Level 2 - Triangular solve
    fn strsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        a: *const f32,
        lda: *const i32,
        x: *mut f32,
        incx: *const i32,
    );
    fn dtrsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        a: *const f64,
        lda: *const i32,
        x: *mut f64,
        incx: *const i32,
    );
    fn ctrsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        a: *const (),
        lda: *const i32,
        x: *mut (),
        incx: *const i32,
    );
    fn ztrsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        a: *const (),
        lda: *const i32,
        x: *mut (),
        incx: *const i32,
    );

    // BLAS Level 2 - Triangular band matrix-vector multiply
    fn stbmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        k: *const i32,
        a: *const f32,
        lda: *const i32,
        x: *mut f32,
        incx: *const i32,
    );
    fn dtbmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        k: *const i32,
        a: *const f64,
        lda: *const i32,
        x: *mut f64,
        incx: *const i32,
    );
    fn ctbmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        k: *const i32,
        a: *const (),
        lda: *const i32,
        x: *mut (),
        incx: *const i32,
    );
    fn ztbmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        k: *const i32,
        a: *const (),
        lda: *const i32,
        x: *mut (),
        incx: *const i32,
    );

    // BLAS Level 2 - Triangular band solve
    fn stbsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        k: *const i32,
        a: *const f32,
        lda: *const i32,
        x: *mut f32,
        incx: *const i32,
    );
    fn dtbsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        k: *const i32,
        a: *const f64,
        lda: *const i32,
        x: *mut f64,
        incx: *const i32,
    );
    fn ctbsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        k: *const i32,
        a: *const (),
        lda: *const i32,
        x: *mut (),
        incx: *const i32,
    );
    fn ztbsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        k: *const i32,
        a: *const (),
        lda: *const i32,
        x: *mut (),
        incx: *const i32,
    );

    // BLAS Level 2 - Rank-1 update (GER)
    fn sger_(
        m: *const i32,
        n: *const i32,
        alpha: *const f32,
        x: *const f32,
        incx: *const i32,
        y: *const f32,
        incy: *const i32,
        a: *mut f32,
        lda: *const i32,
    );
    fn dger_(
        m: *const i32,
        n: *const i32,
        alpha: *const f64,
        x: *const f64,
        incx: *const i32,
        y: *const f64,
        incy: *const i32,
        a: *mut f64,
        lda: *const i32,
    );
    fn cgeru_(
        m: *const i32,
        n: *const i32,
        alpha: *const (),
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
        a: *mut (),
        lda: *const i32,
    );
    fn cgerc_(
        m: *const i32,
        n: *const i32,
        alpha: *const (),
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
        a: *mut (),
        lda: *const i32,
    );
    fn zgeru_(
        m: *const i32,
        n: *const i32,
        alpha: *const (),
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
        a: *mut (),
        lda: *const i32,
    );
    fn zgerc_(
        m: *const i32,
        n: *const i32,
        alpha: *const (),
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
        a: *mut (),
        lda: *const i32,
    );

    // BLAS Level 2 - Symmetric/Hermitian rank-1 update (SYR/HER)
    fn ssyr_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f32,
        x: *const f32,
        incx: *const i32,
        a: *mut f32,
        lda: *const i32,
    );
    fn dsyr_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f64,
        x: *const f64,
        incx: *const i32,
        a: *mut f64,
        lda: *const i32,
    );
    fn cher_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f32,
        x: *const (),
        incx: *const i32,
        a: *mut (),
        lda: *const i32,
    );
    fn zher_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f64,
        x: *const (),
        incx: *const i32,
        a: *mut (),
        lda: *const i32,
    );

    // BLAS Level 2 - Symmetric/Hermitian rank-2 update (SYR2/HER2)
    fn ssyr2_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f32,
        x: *const f32,
        incx: *const i32,
        y: *const f32,
        incy: *const i32,
        a: *mut f32,
        lda: *const i32,
    );
    fn dsyr2_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f64,
        x: *const f64,
        incx: *const i32,
        y: *const f64,
        incy: *const i32,
        a: *mut f64,
        lda: *const i32,
    );
    fn cher2_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const (),
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
        a: *mut (),
        lda: *const i32,
    );
    fn zher2_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const (),
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
        a: *mut (),
        lda: *const i32,
    );

    // BLAS Level 2 - Symmetric/Hermitian packed matrix-vector multiply
    fn sspmv_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f32,
        ap: *const f32,
        x: *const f32,
        incx: *const i32,
        beta: *const f32,
        y: *mut f32,
        incy: *const i32,
    );
    fn dspmv_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f64,
        ap: *const f64,
        x: *const f64,
        incx: *const i32,
        beta: *const f64,
        y: *mut f64,
        incy: *const i32,
    );
    fn chpmv_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const (),
        ap: *const (),
        x: *const (),
        incx: *const i32,
        beta: *const (),
        y: *mut (),
        incy: *const i32,
    );
    fn zhpmv_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const (),
        ap: *const (),
        x: *const (),
        incx: *const i32,
        beta: *const (),
        y: *mut (),
        incy: *const i32,
    );

    // BLAS Level 2 - Triangular packed matrix-vector multiply
    fn stpmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        ap: *const f32,
        x: *mut f32,
        incx: *const i32,
    );
    fn dtpmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        ap: *const f64,
        x: *mut f64,
        incx: *const i32,
    );
    fn ctpmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        ap: *const (),
        x: *mut (),
        incx: *const i32,
    );
    fn ztpmv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        ap: *const (),
        x: *mut (),
        incx: *const i32,
    );

    // BLAS Level 2 - Triangular packed solve
    fn stpsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        ap: *const f32,
        x: *mut f32,
        incx: *const i32,
    );
    fn dtpsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        ap: *const f64,
        x: *mut f64,
        incx: *const i32,
    );
    fn ctpsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        ap: *const (),
        x: *mut (),
        incx: *const i32,
    );
    fn ztpsv_(
        uplo: *const i8,
        trans: *const i8,
        diag: *const i8,
        n: *const i32,
        ap: *const (),
        x: *mut (),
        incx: *const i32,
    );

    // BLAS Level 2 - Symmetric/Hermitian packed rank-1 update
    fn sspr_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f32,
        x: *const f32,
        incx: *const i32,
        ap: *mut f32,
    );
    fn dspr_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f64,
        x: *const f64,
        incx: *const i32,
        ap: *mut f64,
    );
    fn chpr_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f32,
        x: *const (),
        incx: *const i32,
        ap: *mut (),
    );
    fn zhpr_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f64,
        x: *const (),
        incx: *const i32,
        ap: *mut (),
    );

    // BLAS Level 2 - Symmetric/Hermitian packed rank-2 update
    fn sspr2_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f32,
        x: *const f32,
        incx: *const i32,
        y: *const f32,
        incy: *const i32,
        ap: *mut f32,
    );
    fn dspr2_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const f64,
        x: *const f64,
        incx: *const i32,
        y: *const f64,
        incy: *const i32,
        ap: *mut f64,
    );
    fn chpr2_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const (),
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
        ap: *mut (),
    );
    fn zhpr2_(
        uplo: *const i8,
        n: *const i32,
        alpha: *const (),
        x: *const (),
        incx: *const i32,
        y: *const (),
        incy: *const i32,
        ap: *mut (),
    );
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

        // BLAS Level 2 - GEMV
        register_sgemv(std::mem::transmute(sgemv_ as *const ()));
        register_dgemv(std::mem::transmute(dgemv_ as *const ()));
        register_cgemv(std::mem::transmute(cgemv_ as *const ()));
        register_zgemv(std::mem::transmute(zgemv_ as *const ()));

        // BLAS Level 2 - GBMV
        register_sgbmv(std::mem::transmute(sgbmv_ as *const ()));
        register_dgbmv(std::mem::transmute(dgbmv_ as *const ()));
        register_cgbmv(std::mem::transmute(cgbmv_ as *const ()));
        register_zgbmv(std::mem::transmute(zgbmv_ as *const ()));

        // BLAS Level 2 - SYMV/HEMV
        register_ssymv(std::mem::transmute(ssymv_ as *const ()));
        register_dsymv(std::mem::transmute(dsymv_ as *const ()));
        register_chemv(std::mem::transmute(chemv_ as *const ()));
        register_zhemv(std::mem::transmute(zhemv_ as *const ()));

        // BLAS Level 2 - SBMV/HBMV (symmetric/hermitian band matrix-vector)
        register_ssbmv(std::mem::transmute(ssbmv_ as *const ()));
        register_dsbmv(std::mem::transmute(dsbmv_ as *const ()));
        register_chbmv(std::mem::transmute(chbmv_ as *const ()));
        register_zhbmv(std::mem::transmute(zhbmv_ as *const ()));

        // BLAS Level 2 - TRMV
        register_strmv(std::mem::transmute(strmv_ as *const ()));
        register_dtrmv(std::mem::transmute(dtrmv_ as *const ()));
        register_ctrmv(std::mem::transmute(ctrmv_ as *const ()));
        register_ztrmv(std::mem::transmute(ztrmv_ as *const ()));

        // BLAS Level 2 - TRSV
        register_strsv(std::mem::transmute(strsv_ as *const ()));
        register_dtrsv(std::mem::transmute(dtrsv_ as *const ()));
        register_ctrsv(std::mem::transmute(ctrsv_ as *const ()));
        register_ztrsv(std::mem::transmute(ztrsv_ as *const ()));

        // BLAS Level 2 - TBMV (triangular band matrix-vector multiply)
        register_stbmv(std::mem::transmute(stbmv_ as *const ()));
        register_dtbmv(std::mem::transmute(dtbmv_ as *const ()));
        register_ctbmv(std::mem::transmute(ctbmv_ as *const ()));
        register_ztbmv(std::mem::transmute(ztbmv_ as *const ()));

        // BLAS Level 2 - TBSV (triangular band solve)
        register_stbsv(std::mem::transmute(stbsv_ as *const ()));
        register_dtbsv(std::mem::transmute(dtbsv_ as *const ()));
        register_ctbsv(std::mem::transmute(ctbsv_ as *const ()));
        register_ztbsv(std::mem::transmute(ztbsv_ as *const ()));

        // BLAS Level 2 - GER (rank-1 update)
        register_sger(std::mem::transmute(sger_ as *const ()));
        register_dger(std::mem::transmute(dger_ as *const ()));
        register_cgeru(std::mem::transmute(cgeru_ as *const ()));
        register_cgerc(std::mem::transmute(cgerc_ as *const ()));
        register_zgeru(std::mem::transmute(zgeru_ as *const ()));
        register_zgerc(std::mem::transmute(zgerc_ as *const ()));

        // BLAS Level 2 - SYR/HER (symmetric/hermitian rank-1 update)
        register_ssyr(std::mem::transmute(ssyr_ as *const ()));
        register_dsyr(std::mem::transmute(dsyr_ as *const ()));
        register_cher(std::mem::transmute(cher_ as *const ()));
        register_zher(std::mem::transmute(zher_ as *const ()));

        // BLAS Level 2 - SYR2/HER2 (symmetric/hermitian rank-2 update)
        register_ssyr2(std::mem::transmute(ssyr2_ as *const ()));
        register_dsyr2(std::mem::transmute(dsyr2_ as *const ()));
        register_cher2(std::mem::transmute(cher2_ as *const ()));
        register_zher2(std::mem::transmute(zher2_ as *const ()));

        // BLAS Level 2 - SPMV/HPMV (symmetric/hermitian packed matrix-vector multiply)
        register_sspmv(std::mem::transmute(sspmv_ as *const ()));
        register_dspmv(std::mem::transmute(dspmv_ as *const ()));
        register_chpmv(std::mem::transmute(chpmv_ as *const ()));
        register_zhpmv(std::mem::transmute(zhpmv_ as *const ()));

        // BLAS Level 2 - TPMV (triangular packed matrix-vector multiply)
        register_stpmv(std::mem::transmute(stpmv_ as *const ()));
        register_dtpmv(std::mem::transmute(dtpmv_ as *const ()));
        register_ctpmv(std::mem::transmute(ctpmv_ as *const ()));
        register_ztpmv(std::mem::transmute(ztpmv_ as *const ()));

        // BLAS Level 2 - TPSV (triangular packed solve)
        register_stpsv(std::mem::transmute(stpsv_ as *const ()));
        register_dtpsv(std::mem::transmute(dtpsv_ as *const ()));
        register_ctpsv(std::mem::transmute(ctpsv_ as *const ()));
        register_ztpsv(std::mem::transmute(ztpsv_ as *const ()));

        // BLAS Level 2 - SPR/HPR (symmetric/hermitian packed rank-1 update)
        register_sspr(std::mem::transmute(sspr_ as *const ()));
        register_dspr(std::mem::transmute(dspr_ as *const ()));
        register_chpr(std::mem::transmute(chpr_ as *const ()));
        register_zhpr(std::mem::transmute(zhpr_ as *const ()));

        // BLAS Level 2 - SPR2/HPR2 (symmetric/hermitian packed rank-2 update)
        register_sspr2(std::mem::transmute(sspr2_ as *const ()));
        register_dspr2(std::mem::transmute(dspr2_ as *const ()));
        register_chpr2(std::mem::transmute(chpr2_ as *const ()));
        register_zhpr2(std::mem::transmute(zhpr2_ as *const ()));

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
