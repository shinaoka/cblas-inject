//! Benchmark to measure trampoline overhead for BLAS Level 1 operations.
//!
//! Compares:
//! 1. Direct Fortran BLAS call (baseline)
//! 2. cblas-trampoline call (measures overhead)
//! 3. Pure Rust implementation (for reference)

use std::ffi::c_char;
use std::hint::black_box;
use std::time::{Duration, Instant};

// Link against OpenBLAS
#[link(name = "openblas")]
extern "C" {
    // Fortran BLAS (column-major, pass by reference)
    fn daxpy_(
        n: *const i32,
        alpha: *const f64,
        x: *const f64,
        incx: *const i32,
        y: *mut f64,
        incy: *const i32,
    );

    fn ddot_(
        n: *const i32,
        x: *const f64,
        incx: *const i32,
        y: *const f64,
        incy: *const i32,
    ) -> f64;

    fn dnrm2_(n: *const i32, x: *const f64, incx: *const i32) -> f64;

    fn dscal_(n: *const i32, alpha: *const f64, x: *mut f64, incx: *const i32);

    // CBLAS (for comparison)
    fn cblas_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32);

    fn cblas_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64;

    fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64;

    fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32);
}

// Trampoline types matching cblas-trampoline
type DaxpyFnPtr = unsafe extern "C" fn(
    n: *const i32,
    alpha: *const f64,
    x: *const f64,
    incx: *const i32,
    y: *mut f64,
    incy: *const i32,
);

type DdotFnPtr = unsafe extern "C" fn(
    n: *const i32,
    x: *const f64,
    incx: *const i32,
    y: *const f64,
    incy: *const i32,
) -> f64;

type Dnrm2FnPtr = unsafe extern "C" fn(n: *const i32, x: *const f64, incx: *const i32) -> f64;

type DscalFnPtr =
    unsafe extern "C" fn(n: *const i32, alpha: *const f64, x: *mut f64, incx: *const i32);

// Simulated trampoline (function pointer indirection)
static mut DAXPY_PTR: Option<DaxpyFnPtr> = None;
static mut DDOT_PTR: Option<DdotFnPtr> = None;
static mut DNRM2_PTR: Option<Dnrm2FnPtr> = None;
static mut DSCAL_PTR: Option<DscalFnPtr> = None;

unsafe fn trampoline_daxpy(n: i32, alpha: f64, x: *const f64, incx: i32, y: *mut f64, incy: i32) {
    let f = DAXPY_PTR.unwrap();
    f(&n, &alpha, x, &incx, y, &incy);
}

unsafe fn trampoline_ddot(n: i32, x: *const f64, incx: i32, y: *const f64, incy: i32) -> f64 {
    let f = DDOT_PTR.unwrap();
    f(&n, x, &incx, y, &incy)
}

unsafe fn trampoline_dnrm2(n: i32, x: *const f64, incx: i32) -> f64 {
    let f = DNRM2_PTR.unwrap();
    f(&n, x, &incx)
}

unsafe fn trampoline_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32) {
    let f = DSCAL_PTR.unwrap();
    f(&n, &alpha, x, &incx)
}

// Pure Rust implementations for reference
fn rust_daxpy(n: usize, alpha: f64, x: &[f64], y: &mut [f64]) {
    for i in 0..n {
        y[i] += alpha * x[i];
    }
}

fn rust_ddot(n: usize, x: &[f64], y: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        sum += x[i] * y[i];
    }
    sum
}

fn rust_dnrm2(n: usize, x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..n {
        sum += x[i] * x[i];
    }
    sum.sqrt()
}

fn rust_dscal(n: usize, alpha: f64, x: &mut [f64]) {
    for i in 0..n {
        x[i] *= alpha;
    }
}

fn benchmark<F>(name: &str, iterations: usize, mut f: F) -> Duration
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..100 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();

    elapsed
}

fn main() {
    // Initialize trampoline pointers
    unsafe {
        DAXPY_PTR = Some(std::mem::transmute(daxpy_ as *const ()));
        DDOT_PTR = Some(std::mem::transmute(ddot_ as *const ()));
        DNRM2_PTR = Some(std::mem::transmute(dnrm2_ as *const ()));
        DSCAL_PTR = Some(std::mem::transmute(dscal_ as *const ()));
    }

    let sizes = [10, 100, 1000, 10000, 100000, 1000000];
    let iterations = 10000;

    println!("BLAS Level 1 Trampoline Overhead Benchmark");
    println!("==========================================");
    println!("Iterations per measurement: {}", iterations);
    println!();

    for &n in &sizes {
        let mut x: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
        let mut y: Vec<f64> = (0..n).map(|i| (n - i) as f64 * 0.001).collect();
        let alpha = 2.5;
        let n_i32 = n as i32;
        let inc: i32 = 1;

        println!("n = {}", n);
        println!("---------");

        // DAXPY benchmark
        {
            let mut y_copy = y.clone();
            let direct = benchmark("direct", iterations, || unsafe {
                daxpy_(&n_i32, &alpha, x.as_ptr(), &inc, y_copy.as_mut_ptr(), &inc);
                black_box(&y_copy);
            });

            y_copy = y.clone();
            let cblas = benchmark("cblas", iterations, || unsafe {
                cblas_daxpy(n_i32, alpha, x.as_ptr(), inc, y_copy.as_mut_ptr(), inc);
                black_box(&y_copy);
            });

            y_copy = y.clone();
            let trampoline = benchmark("trampoline", iterations, || unsafe {
                trampoline_daxpy(n_i32, alpha, x.as_ptr(), inc, y_copy.as_mut_ptr(), inc);
                black_box(&y_copy);
            });

            y_copy = y.clone();
            let rust = benchmark("rust", iterations, || {
                rust_daxpy(n, alpha, &x, &mut y_copy);
                black_box(&y_copy);
            });

            let direct_ns = direct.as_nanos() as f64 / iterations as f64;
            let cblas_ns = cblas.as_nanos() as f64 / iterations as f64;
            let trampoline_ns = trampoline.as_nanos() as f64 / iterations as f64;
            let rust_ns = rust.as_nanos() as f64 / iterations as f64;

            println!(
                "  DAXPY: direct={:.1}ns, cblas={:.1}ns, trampoline={:.1}ns, rust={:.1}ns",
                direct_ns, cblas_ns, trampoline_ns, rust_ns
            );
            println!(
                "         overhead: cblas={:.1}%, trampoline={:.1}%",
                (cblas_ns - direct_ns) / direct_ns * 100.0,
                (trampoline_ns - direct_ns) / direct_ns * 100.0
            );
        }

        // DDOT benchmark
        {
            let direct = benchmark("direct", iterations, || unsafe {
                let result = ddot_(&n_i32, x.as_ptr(), &inc, y.as_ptr(), &inc);
                black_box(result);
            });

            let cblas = benchmark("cblas", iterations, || unsafe {
                let result = cblas_ddot(n_i32, x.as_ptr(), inc, y.as_ptr(), inc);
                black_box(result);
            });

            let trampoline = benchmark("trampoline", iterations, || unsafe {
                let result = trampoline_ddot(n_i32, x.as_ptr(), inc, y.as_ptr(), inc);
                black_box(result);
            });

            let rust = benchmark("rust", iterations, || {
                let result = rust_ddot(n, &x, &y);
                black_box(result);
            });

            let direct_ns = direct.as_nanos() as f64 / iterations as f64;
            let cblas_ns = cblas.as_nanos() as f64 / iterations as f64;
            let trampoline_ns = trampoline.as_nanos() as f64 / iterations as f64;
            let rust_ns = rust.as_nanos() as f64 / iterations as f64;

            println!(
                "  DDOT:  direct={:.1}ns, cblas={:.1}ns, trampoline={:.1}ns, rust={:.1}ns",
                direct_ns, cblas_ns, trampoline_ns, rust_ns
            );
            println!(
                "         overhead: cblas={:.1}%, trampoline={:.1}%",
                (cblas_ns - direct_ns) / direct_ns * 100.0,
                (trampoline_ns - direct_ns) / direct_ns * 100.0
            );
        }

        // DNRM2 benchmark
        {
            let direct = benchmark("direct", iterations, || unsafe {
                let result = dnrm2_(&n_i32, x.as_ptr(), &inc);
                black_box(result);
            });

            let cblas = benchmark("cblas", iterations, || unsafe {
                let result = cblas_dnrm2(n_i32, x.as_ptr(), inc);
                black_box(result);
            });

            let trampoline = benchmark("trampoline", iterations, || unsafe {
                let result = trampoline_dnrm2(n_i32, x.as_ptr(), inc);
                black_box(result);
            });

            let rust = benchmark("rust", iterations, || {
                let result = rust_dnrm2(n, &x);
                black_box(result);
            });

            let direct_ns = direct.as_nanos() as f64 / iterations as f64;
            let cblas_ns = cblas.as_nanos() as f64 / iterations as f64;
            let trampoline_ns = trampoline.as_nanos() as f64 / iterations as f64;
            let rust_ns = rust.as_nanos() as f64 / iterations as f64;

            println!(
                "  DNRM2: direct={:.1}ns, cblas={:.1}ns, trampoline={:.1}ns, rust={:.1}ns",
                direct_ns, cblas_ns, trampoline_ns, rust_ns
            );
            println!(
                "         overhead: cblas={:.1}%, trampoline={:.1}%",
                (cblas_ns - direct_ns) / direct_ns * 100.0,
                (trampoline_ns - direct_ns) / direct_ns * 100.0
            );
        }

        // DSCAL benchmark
        {
            let mut x_copy = x.clone();
            let direct = benchmark("direct", iterations, || unsafe {
                dscal_(&n_i32, &alpha, x_copy.as_mut_ptr(), &inc);
                black_box(&x_copy);
            });

            x_copy = x.clone();
            let cblas = benchmark("cblas", iterations, || unsafe {
                cblas_dscal(n_i32, alpha, x_copy.as_mut_ptr(), inc);
                black_box(&x_copy);
            });

            x_copy = x.clone();
            let trampoline = benchmark("trampoline", iterations, || unsafe {
                trampoline_dscal(n_i32, alpha, x_copy.as_mut_ptr(), inc);
                black_box(&x_copy);
            });

            x_copy = x.clone();
            let rust = benchmark("rust", iterations, || {
                rust_dscal(n, alpha, &mut x_copy);
                black_box(&x_copy);
            });

            let direct_ns = direct.as_nanos() as f64 / iterations as f64;
            let cblas_ns = cblas.as_nanos() as f64 / iterations as f64;
            let trampoline_ns = trampoline.as_nanos() as f64 / iterations as f64;
            let rust_ns = rust.as_nanos() as f64 / iterations as f64;

            println!(
                "  DSCAL: direct={:.1}ns, cblas={:.1}ns, trampoline={:.1}ns, rust={:.1}ns",
                direct_ns, cblas_ns, trampoline_ns, rust_ns
            );
            println!(
                "         overhead: cblas={:.1}%, trampoline={:.1}%",
                (cblas_ns - direct_ns) / direct_ns * 100.0,
                (trampoline_ns - direct_ns) / direct_ns * 100.0
            );
        }

        println!();
    }
}
