#![cfg(all(unix, not(feature = "openblas")))]

use cblas_inject::{
    cblas_cgemm, cblas_dgemm, cblas_inject_register_cgemm_lp64, cblas_inject_register_dgemm_lp64,
    cblas_inject_register_sgemm_lp64, cblas_inject_register_zgemm_lp64, cblas_sgemm, cblas_zgemm,
    CblasColMajor, CblasNoTrans, CblasRowMajor,
};
use num_complex::{Complex32, Complex64};
use std::ffi::c_char;
use std::sync::atomic::{AtomicPtr, Ordering};

static PROTECTED_BASE: AtomicPtr<libc::c_void> = AtomicPtr::new(std::ptr::null_mut());
static PAGE_SIZE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn page_size() -> usize {
    let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    assert!(size > 0, "sysconf(_SC_PAGESIZE) failed: {size}");
    size as usize
}

unsafe fn make_readable(_c: *mut libc::c_void) -> bool {
    let address = PROTECTED_BASE.load(Ordering::Relaxed);
    if address.is_null() {
        return true;
    }
    let page = PAGE_SIZE.load(Ordering::Relaxed);
    if page == 0 || libc::mprotect(address, page, libc::PROT_READ | libc::PROT_WRITE) != 0 {
        libc::abort();
    }
    true
}

// These callbacks intentionally mirror the 13-argument BLAS ABI.
#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn sgemm_contract(
    ta: *const c_char,
    tb: *const c_char,
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
) {
    make_readable(c.cast());
    gemm_loop(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn dgemm_contract(
    ta: *const c_char,
    tb: *const c_char,
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
) {
    make_readable(c.cast());
    gemm_loop(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn cgemm_contract(
    ta: *const c_char,
    tb: *const c_char,
    m: *const i32,
    n: *const i32,
    k: *const i32,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: *const i32,
    b: *const Complex32,
    ldb: *const i32,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: *const i32,
) {
    make_readable(c.cast());
    gemm_loop(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn zgemm_contract(
    ta: *const c_char,
    tb: *const c_char,
    m: *const i32,
    n: *const i32,
    k: *const i32,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: *const i32,
    b: *const Complex64,
    ldb: *const i32,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const i32,
) {
    make_readable(c.cast());
    gemm_loop(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

unsafe fn protected_page() -> *mut libc::c_void {
    // This guard page proves only that the wrapper does not read C before the
    // callback runs; semantic poison tests cover the repository callbacks,
    // while arbitrary external callback behavior remains the registration
    // contract of the unsafe API.
    let page = page_size();
    PAGE_SIZE.store(page, Ordering::Relaxed);
    let p = libc::mmap(
        std::ptr::null_mut(),
        page,
        libc::PROT_NONE,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
        -1,
        0,
    );
    assert_ne!(p, libc::MAP_FAILED);
    assert_eq!((p as usize) % page, 0);
    PROTECTED_BASE.store(p, Ordering::Relaxed);
    p
}

unsafe fn release_page(p: *mut libc::c_void) {
    assert_eq!(libc::munmap(p, PAGE_SIZE.load(Ordering::Relaxed)), 0);
    PROTECTED_BASE.store(std::ptr::null_mut(), Ordering::Relaxed);
}

trait Scalar: Copy {
    fn zero() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn conj(self) -> Self;
}
impl Scalar for f32 {
    fn zero() -> Self {
        0.0
    }
    fn add(self, r: Self) -> Self {
        self + r
    }
    fn mul(self, r: Self) -> Self {
        self * r
    }
    fn conj(self) -> Self {
        self
    }
}
impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn add(self, r: Self) -> Self {
        self + r
    }
    fn mul(self, r: Self) -> Self {
        self * r
    }
    fn conj(self) -> Self {
        self
    }
}
impl Scalar for Complex32 {
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    fn add(self, r: Self) -> Self {
        self + r
    }
    fn mul(self, r: Self) -> Self {
        self * r
    }
    fn conj(self) -> Self {
        Complex32::new(self.re, -self.im)
    }
}
impl Scalar for Complex64 {
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    fn add(self, r: Self) -> Self {
        self + r
    }
    fn mul(self, r: Self) -> Self {
        self * r
    }
    fn conj(self) -> Self {
        Complex64::new(self.re, -self.im)
    }
}

unsafe fn gemm_loop<T: Scalar + PartialEq, I: Copy + Into<i64>>(
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
    let (m, n, k) = (
        (*m).into() as usize,
        (*n).into() as usize,
        (*k).into() as usize,
    );
    let (lda, ldb, ldc) = (
        (*lda).into() as usize,
        (*ldb).into() as usize,
        (*ldc).into() as usize,
    );
    let trans = |flag: c_char| flag as u8 as char;
    let av = |row: usize, col: usize| {
        let flag = trans(*ta);
        let (r, q) = if flag == 'N' { (row, col) } else { (col, row) };
        let mut v = *a.add(r + q * lda);
        if flag == 'C' {
            v = v.conj();
        }
        v
    };
    let bv = |row: usize, col: usize| {
        let flag = trans(*tb);
        let (r, q) = if flag == 'N' { (row, col) } else { (col, row) };
        let mut v = *b.add(r + q * ldb);
        if flag == 'C' {
            v = v.conj();
        }
        v
    };
    for col in 0..n {
        for row in 0..m {
            let mut sum = T::zero();
            for q in 0..k {
                sum = sum.add(av(row, q).mul(bv(q, col)));
            }
            let value = (*alpha).mul(sum);
            let slot = c.add(row + col * ldc);
            *slot = if *beta == T::zero() {
                value
            } else {
                value.add((*beta).mul(*slot))
            };
        }
    }
}

#[test]
fn all_gemm_families_preserve_beta_contract_in_both_orders() {
    unsafe {
        assert_eq!(
            cblas_inject_register_sgemm_lp64(sgemm_contract as *const _),
            0
        );
        assert_eq!(
            cblas_inject_register_dgemm_lp64(dgemm_contract as *const _),
            0
        );
        assert_eq!(
            cblas_inject_register_cgemm_lp64(cgemm_contract as *const _),
            0
        );
        assert_eq!(
            cblas_inject_register_zgemm_lp64(zgemm_contract as *const _),
            0
        );
        for &order in &[CblasColMajor, CblasRowMajor] {
            let c = protected_page();
            cblas_sgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                2.0,
                &3.0,
                1,
                &4.0,
                1,
                0.0,
                c.cast(),
                1,
            );
            assert_eq!(*(c.cast::<f32>()), 24.0);
            release_page(c);
            let c = protected_page();
            cblas_dgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                2.0,
                &3.0,
                1,
                &4.0,
                1,
                0.0,
                c.cast(),
                1,
            );
            assert_eq!(*(c.cast::<f64>()), 24.0);
            release_page(c);
            let c = protected_page();
            let a = Complex32::new(2.0, 1.0);
            let b = Complex32::new(3.0, -1.0);
            let alpha = Complex32::new(1.0, 0.0);
            let beta = Complex32::new(0.0, 0.0);
            cblas_cgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                &alpha,
                &a,
                1,
                &b,
                1,
                &beta,
                c.cast(),
                1,
            );
            assert_eq!(*(c.cast::<Complex32>()), a * b);
            release_page(c);
            let c = protected_page();
            let a = Complex64::new(2.0, 1.0);
            let b = Complex64::new(3.0, -1.0);
            let alpha = Complex64::new(1.0, 0.0);
            let beta = Complex64::new(0.0, 0.0);
            cblas_zgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                &alpha,
                &a,
                1,
                &b,
                1,
                &beta,
                c.cast(),
                1,
            );
            assert_eq!(*(c.cast::<Complex64>()), a * b);
            release_page(c);
        }
        // Semantic poison: a callback that evaluates `beta * old_c` even for
        // exact-zero beta would propagate these NaNs. This checks the
        // repository callbacks; arbitrary external callbacks remain an unsafe
        // registration obligation documented by the API.
        for &order in &[CblasColMajor, CblasRowMajor] {
            let mut s = f32::NAN;
            cblas_sgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                2.0,
                &3.0,
                1,
                &4.0,
                1,
                0.0,
                &mut s,
                1,
            );
            assert_eq!(s, 24.0);
            let mut d = f64::NAN;
            cblas_dgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                2.0,
                &3.0,
                1,
                &4.0,
                1,
                0.0,
                &mut d,
                1,
            );
            assert_eq!(d, 24.0);
            let a = Complex32::new(2.0, 1.0);
            let b = Complex32::new(3.0, -1.0);
            let alpha = Complex32::new(1.0, 0.0);
            let beta = Complex32::new(0.0, 0.0);
            let mut c = Complex32::new(f32::NAN, f32::NAN);
            cblas_cgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                &alpha,
                &a,
                1,
                &b,
                1,
                &beta,
                &mut c,
                1,
            );
            assert_eq!(c, a * b);
            let a = Complex64::new(2.0, 1.0);
            let b = Complex64::new(3.0, -1.0);
            let alpha = Complex64::new(1.0, 0.0);
            let beta = Complex64::new(0.0, 0.0);
            let mut c = Complex64::new(f64::NAN, f64::NAN);
            cblas_zgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                &alpha,
                &a,
                1,
                &b,
                1,
                &beta,
                &mut c,
                1,
            );
            assert_eq!(c, a * b);
        }
        for &order in &[CblasColMajor, CblasRowMajor] {
            let mut s = 5.0;
            cblas_sgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                2.0,
                &3.0,
                1,
                &4.0,
                1,
                0.5,
                &mut s,
                1,
            );
            assert_eq!(s, 26.5);
            let mut d = 5.0;
            cblas_dgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                2.0,
                &3.0,
                1,
                &4.0,
                1,
                0.5,
                &mut d,
                1,
            );
            assert_eq!(d, 26.5);
            let a = Complex32::new(2.0, 1.0);
            let b = Complex32::new(3.0, -1.0);
            let alpha = Complex32::new(1.0, 0.0);
            let beta = Complex32::new(0.5, 0.0);
            let mut c = Complex32::new(5.0, 0.0);
            cblas_cgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                &alpha,
                &a,
                1,
                &b,
                1,
                &beta,
                &mut c,
                1,
            );
            assert_eq!(c, a * b + beta * Complex32::new(5.0, 0.0));
            let a = Complex64::new(2.0, 1.0);
            let b = Complex64::new(3.0, -1.0);
            let alpha = Complex64::new(1.0, 0.0);
            let beta = Complex64::new(0.5, 0.0);
            let mut c = Complex64::new(5.0, 0.0);
            cblas_zgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                1,
                1,
                1,
                &alpha,
                &a,
                1,
                &b,
                1,
                &beta,
                &mut c,
                1,
            );
            assert_eq!(c, a * b + beta * Complex64::new(5.0, 0.0));
            nontrivial_and_zero_k_cover_every_logical_element_and_preserve_padding();
        }
    }
}

fn nontrivial_and_zero_k_cover_every_logical_element_and_preserve_padding() {
    unsafe {
        for &order in &[CblasColMajor, CblasRowMajor] {
            let a_s = [1.0f32; 12];
            let b_s = [1.0f32; 12];
            let mut c_s = [f32::NAN; 12];
            cblas_sgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                2,
                2.0,
                a_s.as_ptr(),
                3,
                b_s.as_ptr(),
                4,
                0.0,
                c_s.as_mut_ptr(),
                4,
            );
            for col in 0..3 {
                for row in 0..2 {
                    let i = if order == CblasColMajor {
                        row + col * 4
                    } else {
                        row * 4 + col
                    };
                    assert_eq!(c_s[i], 4.0);
                }
                let p = if order == CblasColMajor {
                    2 + col * 4
                } else {
                    col * 4 + 3
                };
                assert!(c_s[p].is_nan());
            }
            let a_d = [1.0f64; 12];
            let b_d = [1.0f64; 12];
            let mut c_d = [f64::NAN; 12];
            cblas_dgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                2,
                2.0,
                a_d.as_ptr(),
                3,
                b_d.as_ptr(),
                4,
                0.0,
                c_d.as_mut_ptr(),
                4,
            );
            for col in 0..3 {
                for row in 0..2 {
                    let i = if order == CblasColMajor {
                        row + col * 4
                    } else {
                        row * 4 + col
                    };
                    assert_eq!(c_d[i], 4.0);
                }
                let p = if order == CblasColMajor {
                    2 + col * 4
                } else {
                    col * 4 + 3
                };
                assert!(c_d[p].is_nan());
            }
            let a_c = [Complex32::new(1.0, 0.0); 12];
            let b_c = [Complex32::new(1.0, 0.0); 12];
            let mut c_c = [Complex32::new(f32::NAN, f32::NAN); 12];
            cblas_cgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                2,
                &Complex32::new(2.0, 0.0),
                a_c.as_ptr(),
                3,
                b_c.as_ptr(),
                4,
                &Complex32::new(0.0, 0.0),
                c_c.as_mut_ptr(),
                4,
            );
            for col in 0..3 {
                for row in 0..2 {
                    let i = if order == CblasColMajor {
                        row + col * 4
                    } else {
                        row * 4 + col
                    };
                    assert_eq!(c_c[i], Complex32::new(4.0, 0.0));
                }
                let p = if order == CblasColMajor {
                    2 + col * 4
                } else {
                    col * 4 + 3
                };
                assert!(c_c[p].re.is_nan());
            }
            let a_z = [Complex64::new(1.0, 0.0); 12];
            let b_z = [Complex64::new(1.0, 0.0); 12];
            let mut c_z = [Complex64::new(f64::NAN, f64::NAN); 12];
            cblas_zgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                2,
                &Complex64::new(2.0, 0.0),
                a_z.as_ptr(),
                3,
                b_z.as_ptr(),
                4,
                &Complex64::new(0.0, 0.0),
                c_z.as_mut_ptr(),
                4,
            );
            for col in 0..3 {
                for row in 0..2 {
                    let i = if order == CblasColMajor {
                        row + col * 4
                    } else {
                        row * 4 + col
                    };
                    assert_eq!(c_z[i], Complex64::new(4.0, 0.0));
                }
                let p = if order == CblasColMajor {
                    2 + col * 4
                } else {
                    col * 4 + 3
                };
                assert!(c_z[p].re.is_nan());
            }

            let mut zs = [f32::NAN; 12];
            cblas_sgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                0,
                1.0,
                [].as_ptr(),
                3,
                [].as_ptr(),
                4,
                0.0,
                zs.as_mut_ptr(),
                4,
            );
            let mut zd = [f64::NAN; 12];
            cblas_dgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                0,
                1.0,
                [].as_ptr(),
                3,
                [].as_ptr(),
                4,
                0.0,
                zd.as_mut_ptr(),
                4,
            );
            let mut zc = [Complex32::new(f32::NAN, f32::NAN); 12];
            cblas_cgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                0,
                &Complex32::new(1.0, 0.0),
                [].as_ptr(),
                3,
                [].as_ptr(),
                4,
                &Complex32::new(0.0, 0.0),
                zc.as_mut_ptr(),
                4,
            );
            let mut zz = [Complex64::new(f64::NAN, f64::NAN); 12];
            cblas_zgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                0,
                &Complex64::new(1.0, 0.0),
                [].as_ptr(),
                3,
                [].as_ptr(),
                4,
                &Complex64::new(0.0, 0.0),
                zz.as_mut_ptr(),
                4,
            );
            for col in 0..3 {
                for row in 0..2 {
                    let i = if order == CblasColMajor {
                        row + col * 4
                    } else {
                        row * 4 + col
                    };
                    assert_eq!(zs[i], 0.0);
                    assert_eq!(zd[i], 0.0);
                    assert_eq!(zc[i], Complex32::new(0.0, 0.0));
                    assert_eq!(zz[i], Complex64::new(0.0, 0.0));
                }
                let p = if order == CblasColMajor {
                    2 + col * 4
                } else {
                    col * 4 + 3
                };
                assert!(zs[p].is_nan());
                assert!(zd[p].is_nan());
                assert!(zc[p].re.is_nan());
                assert!(zz[p].re.is_nan());
            }
        }
    }
}
