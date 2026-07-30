#![cfg(feature = "openblas")]

use cblas_inject::{
    cblas_cgemm, cblas_dgemm, cblas_sgemm, cblas_zgemm, is_cgemm_registered, is_dgemm_registered,
    is_sgemm_registered, is_zgemm_registered, CblasColMajor, CblasNoTrans, CblasRowMajor,
    CBLAS_ORDER,
};
use num_complex::{Complex32, Complex64};

fn is_padding(order: CBLAS_ORDER, index: usize) -> bool {
    let row = if order == CblasColMajor {
        index % 4
    } else {
        index / 4
    };
    let col = if order == CblasColMajor {
        index / 4
    } else {
        index % 4
    };
    row >= 2 || col >= 3
}

fn assert_real_padding_f32(order: CBLAS_ORDER, before: &[f32; 12], after: &[f32; 12]) {
    for i in 0..12 {
        if is_padding(order, i) {
            assert_eq!(before[i].to_bits(), after[i].to_bits());
        }
    }
}
fn assert_real_padding_f64(order: CBLAS_ORDER, before: &[f64; 12], after: &[f64; 12]) {
    for i in 0..12 {
        if is_padding(order, i) {
            assert_eq!(before[i].to_bits(), after[i].to_bits());
        }
    }
}
fn assert_complex_padding_f32(
    order: CBLAS_ORDER,
    before: &[Complex32; 12],
    after: &[Complex32; 12],
) {
    for i in 0..12 {
        if is_padding(order, i) {
            assert_eq!(before[i].re.to_bits(), after[i].re.to_bits());
            assert_eq!(before[i].im.to_bits(), after[i].im.to_bits());
        }
    }
}
fn assert_complex_padding_f64(
    order: CBLAS_ORDER,
    before: &[Complex64; 12],
    after: &[Complex64; 12],
) {
    for i in 0..12 {
        if is_padding(order, i) {
            assert_eq!(before[i].re.to_bits(), after[i].re.to_bits());
            assert_eq!(before[i].im.to_bits(), after[i].im.to_bits());
        }
    }
}

fn sentinel_f32() -> [f32; 12] {
    core::array::from_fn(|i| f32::from_bits(0x7fc0_0001 + i as u32))
}
fn sentinel_f64() -> [f64; 12] {
    core::array::from_fn(|i| f64::from_bits(0x7ff0_0000_0000_0001 + i as u64))
}
fn sentinel_c32() -> [Complex32; 12] {
    core::array::from_fn(|i| {
        Complex32::new(
            f32::from_bits(0x7fc0_0001 + i as u32),
            f32::from_bits(0x7fc1_0001 + i as u32),
        )
    })
}
fn sentinel_c64() -> [Complex64; 12] {
    core::array::from_fn(|i| {
        Complex64::new(
            f64::from_bits(0x7ff0_0000_0000_0001 + i as u64),
            f64::from_bits(0x7ff1_0000_0000_0001 + i as u64),
        )
    })
}

#[test]
fn auto_registered_gemm_beta_zero_and_zero_k() {
    assert!(
        is_sgemm_registered(),
        "OpenBLAS sgemm provider is not registered"
    );
    assert!(
        is_dgemm_registered(),
        "OpenBLAS dgemm provider is not registered"
    );
    assert!(
        is_cgemm_registered(),
        "OpenBLAS cgemm provider is not registered"
    );
    assert!(
        is_zgemm_registered(),
        "OpenBLAS zgemm provider is not registered"
    );
    unsafe {
        for &order in &[CblasColMajor, CblasRowMajor] {
            let a = [
                1.0f32, 0.0, 99.0, 0.0, 1.0, 99.0, 0.0, 0.0, 99.0, 0.0, 0.0, 99.0,
            ];
            let b = [
                2.0f32, 2.0, 2.0, 99.0, 2.0, 2.0, 2.0, 99.0, 2.0, 2.0, 2.0, 99.0,
            ];
            let mut c = sentinel_f32();
            let c_before = c;
            cblas_sgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                2,
                2.0,
                a.as_ptr(),
                3,
                b.as_ptr(),
                4,
                0.0,
                c.as_mut_ptr(),
                4,
            );
            let a = [
                1.0f64, 0.0, 99.0, 0.0, 1.0, 99.0, 0.0, 0.0, 99.0, 0.0, 0.0, 99.0,
            ];
            let b = [
                2.0f64, 2.0, 2.0, 99.0, 2.0, 2.0, 2.0, 99.0, 2.0, 2.0, 2.0, 99.0,
            ];
            let mut d = sentinel_f64();
            let d_before = d;
            cblas_dgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                2,
                2.0,
                a.as_ptr(),
                3,
                b.as_ptr(),
                4,
                0.0,
                d.as_mut_ptr(),
                4,
            );
            for col in 0..3 {
                for row in 0..2 {
                    let i = if order == CblasColMajor {
                        row + col * 4
                    } else {
                        row * 4 + col
                    };
                    assert_eq!(c[i], 4.0);
                    assert_eq!(d[i], 4.0);
                }
            }
            let a = [
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(99.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(1.0, 0.0),
                Complex32::new(99.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(99.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(0.0, 0.0),
                Complex32::new(99.0, 0.0),
            ];
            let b = [
                Complex32::new(2.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(99.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(99.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(2.0, 0.0),
                Complex32::new(99.0, 0.0),
            ];
            let mut z = sentinel_c32();
            let z_before = z;
            cblas_cgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                2,
                &Complex32::new(2.0, 0.0),
                a.as_ptr(),
                3,
                b.as_ptr(),
                4,
                &Complex32::new(0.0, 0.0),
                z.as_mut_ptr(),
                4,
            );
            let a = [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(99.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(99.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(99.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(99.0, 0.0),
            ];
            let b = [
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(99.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(99.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(99.0, 0.0),
            ];
            let mut w = sentinel_c64();
            let w_before = w;
            cblas_zgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                2,
                &Complex64::new(2.0, 0.0),
                a.as_ptr(),
                3,
                b.as_ptr(),
                4,
                &Complex64::new(0.0, 0.0),
                w.as_mut_ptr(),
                4,
            );
            for col in 0..3 {
                for row in 0..2 {
                    let i = if order == CblasColMajor {
                        row + col * 4
                    } else {
                        row * 4 + col
                    };
                    assert_eq!(z[i], Complex32::new(4.0, 0.0));
                    assert_eq!(w[i], Complex64::new(4.0, 0.0));
                }
            }
            assert_real_padding_f32(order, &c_before, &c);
            assert_real_padding_f64(order, &d_before, &d);
            assert_complex_padding_f32(order, &z_before, &z);
            assert_complex_padding_f64(order, &w_before, &w);

            let a = [1.0f32; 1];
            let b = [1.0f32; 1];
            let mut c = sentinel_f32();
            let c_before = c;
            cblas_sgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                0,
                2.0,
                a.as_ptr(),
                3,
                b.as_ptr(),
                4,
                0.0,
                c.as_mut_ptr(),
                4,
            );
            let a = [1.0f64; 1];
            let b = [1.0f64; 1];
            let mut d = sentinel_f64();
            let d_before = d;
            cblas_dgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                0,
                2.0,
                a.as_ptr(),
                3,
                b.as_ptr(),
                4,
                0.0,
                d.as_mut_ptr(),
                4,
            );
            let a = [Complex32::new(1.0, 0.0); 1];
            let b = [Complex32::new(1.0, 0.0); 1];
            let mut z = sentinel_c32();
            let z_before = z;
            cblas_cgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                0,
                &Complex32::new(2.0, 0.0),
                a.as_ptr(),
                3,
                b.as_ptr(),
                4,
                &Complex32::new(0.0, 0.0),
                z.as_mut_ptr(),
                4,
            );
            let a = [Complex64::new(1.0, 0.0); 1];
            let b = [Complex64::new(1.0, 0.0); 1];
            let mut w = sentinel_c64();
            let w_before = w;
            cblas_zgemm(
                order,
                CblasNoTrans,
                CblasNoTrans,
                2,
                3,
                0,
                &Complex64::new(2.0, 0.0),
                a.as_ptr(),
                3,
                b.as_ptr(),
                4,
                &Complex64::new(0.0, 0.0),
                w.as_mut_ptr(),
                4,
            );
            for col in 0..3 {
                for row in 0..2 {
                    let i = if order == CblasColMajor {
                        row + col * 4
                    } else {
                        row * 4 + col
                    };
                    assert_eq!(c[i], 0.0);
                    assert_eq!(d[i], 0.0);
                    assert_eq!(z[i], Complex32::new(0.0, 0.0));
                    assert_eq!(w[i], Complex64::new(0.0, 0.0));
                }
            }
            assert_real_padding_f32(order, &c_before, &c);
            assert_real_padding_f64(order, &d_before, &d);
            assert_complex_padding_f32(order, &z_before, &z);
            assert_complex_padding_f64(order, &w_before, &w);
        }
    }
}
