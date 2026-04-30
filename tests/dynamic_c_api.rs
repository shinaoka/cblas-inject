#![cfg(not(feature = "openblas"))]

use std::ffi::{c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};

use libloading::Library;
use num_complex::Complex64;

const CBLAS_ROW_MAJOR: c_int = 101;
const CBLAS_COL_MAJOR: c_int = 102;
const CBLAS_NO_TRANS: c_int = 111;
const CBLAS_INJECT_STATUS_OK: c_int = 0;

#[cfg(feature = "ilp64")]
type CblasInt = i64;
#[cfg(not(feature = "ilp64"))]
type CblasInt = i32;

#[cfg(feature = "ilp64")]
type ProviderInt = i32;
#[cfg(not(feature = "ilp64"))]
type ProviderInt = i64;

type RegisterGemm = unsafe extern "C" fn(*const c_void) -> c_int;
type CapabilityQuery = unsafe extern "C" fn() -> c_int;
type CblasDgemm = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    CblasInt,
    CblasInt,
    CblasInt,
    f64,
    *const f64,
    CblasInt,
    *const f64,
    CblasInt,
    f64,
    *mut f64,
    CblasInt,
);
type CblasZgemm = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    CblasInt,
    CblasInt,
    CblasInt,
    *const Complex64,
    *const Complex64,
    CblasInt,
    *const Complex64,
    CblasInt,
    *const Complex64,
    *mut Complex64,
    CblasInt,
);
type CblasDgemm64 = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    i64,
    i64,
    i64,
    f64,
    *const f64,
    i64,
    *const f64,
    i64,
    f64,
    *mut f64,
    i64,
);
type CblasZgemm64 = unsafe extern "C" fn(
    c_int,
    c_int,
    c_int,
    i64,
    i64,
    i64,
    *const Complex64,
    *const Complex64,
    i64,
    *const Complex64,
    i64,
    *const Complex64,
    *mut Complex64,
    i64,
);

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

unsafe extern "C" fn mock_dgemm_opposite_width(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const ProviderInt,
    n: *const ProviderInt,
    k: *const ProviderInt,
    _alpha: *const f64,
    _a: *const f64,
    lda: *const ProviderInt,
    _b: *const f64,
    ldb: *const ProviderInt,
    _beta: *const f64,
    c: *mut f64,
    ldc: *const ProviderInt,
) {
    DGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    DGEMM_M.store((*m) as i64, Ordering::SeqCst);
    DGEMM_N.store((*n) as i64, Ordering::SeqCst);
    DGEMM_K.store((*k) as i64, Ordering::SeqCst);
    DGEMM_LDA.store((*lda) as i64, Ordering::SeqCst);
    DGEMM_LDB.store((*ldb) as i64, Ordering::SeqCst);
    DGEMM_LDC.store((*ldc) as i64, Ordering::SeqCst);

    #[cfg(feature = "ilp64")]
    {
        *c = 32.0;
    }
    #[cfg(not(feature = "ilp64"))]
    {
        *c = 64.0;
    }
}

unsafe extern "C" fn mock_zgemm_opposite_width(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const ProviderInt,
    n: *const ProviderInt,
    k: *const ProviderInt,
    alpha: *const Complex64,
    _a: *const Complex64,
    lda: *const ProviderInt,
    _b: *const Complex64,
    ldb: *const ProviderInt,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const ProviderInt,
) {
    ZGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    ZGEMM_M.store((*m) as i64, Ordering::SeqCst);
    ZGEMM_N.store((*n) as i64, Ordering::SeqCst);
    ZGEMM_K.store((*k) as i64, Ordering::SeqCst);
    ZGEMM_LDA.store((*lda) as i64, Ordering::SeqCst);
    ZGEMM_LDB.store((*ldb) as i64, Ordering::SeqCst);
    ZGEMM_LDC.store((*ldc) as i64, Ordering::SeqCst);
    ZGEMM_ALPHA_RE.store((*alpha).re.to_bits(), Ordering::SeqCst);
    ZGEMM_ALPHA_IM.store((*alpha).im.to_bits(), Ordering::SeqCst);
    ZGEMM_BETA_RE.store((*beta).re.to_bits(), Ordering::SeqCst);
    ZGEMM_BETA_IM.store((*beta).im.to_bits(), Ordering::SeqCst);

    #[cfg(feature = "ilp64")]
    {
        *c = Complex64::new(32.0, -32.0);
    }
    #[cfg(not(feature = "ilp64"))]
    {
        *c = Complex64::new(64.0, -64.0);
    }
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_dgemm_ilp64_for_cblas64(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    _alpha: *const f64,
    _a: *const f64,
    lda: *const i64,
    _b: *const f64,
    ldb: *const i64,
    _beta: *const f64,
    c: *mut f64,
    ldc: *const i64,
) {
    DGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    DGEMM_M.store(*m, Ordering::SeqCst);
    DGEMM_N.store(*n, Ordering::SeqCst);
    DGEMM_K.store(*k, Ordering::SeqCst);
    DGEMM_LDA.store(*lda, Ordering::SeqCst);
    DGEMM_LDB.store(*ldb, Ordering::SeqCst);
    DGEMM_LDC.store(*ldc, Ordering::SeqCst);
    *c = 640.0;
}

#[cfg(feature = "ilp64")]
unsafe extern "C" fn mock_zgemm_ilp64_for_cblas64(
    _transa: *const c_char,
    _transb: *const c_char,
    m: *const i64,
    n: *const i64,
    k: *const i64,
    alpha: *const Complex64,
    _a: *const Complex64,
    lda: *const i64,
    _b: *const Complex64,
    ldb: *const i64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: *const i64,
) {
    ZGEMM_CALLS.fetch_add(1, Ordering::SeqCst);
    ZGEMM_M.store(*m, Ordering::SeqCst);
    ZGEMM_N.store(*n, Ordering::SeqCst);
    ZGEMM_K.store(*k, Ordering::SeqCst);
    ZGEMM_LDA.store(*lda, Ordering::SeqCst);
    ZGEMM_LDB.store(*ldb, Ordering::SeqCst);
    ZGEMM_LDC.store(*ldc, Ordering::SeqCst);
    ZGEMM_ALPHA_RE.store((*alpha).re.to_bits(), Ordering::SeqCst);
    ZGEMM_ALPHA_IM.store((*alpha).im.to_bits(), Ordering::SeqCst);
    ZGEMM_BETA_RE.store((*beta).re.to_bits(), Ordering::SeqCst);
    ZGEMM_BETA_IM.store((*beta).im.to_bits(), Ordering::SeqCst);
    *c = Complex64::new(640.0, -640.0);
}

#[test]
fn loaded_cdylib_registers_opposite_width_providers_and_dispatches_unprefixed_gemm() {
    let library_path = find_cdylib();
    let library = unsafe {
        Library::new(&library_path)
            .unwrap_or_else(|err| panic!("failed to load {}: {err}", library_path.display()))
    };

    unsafe {
        let blas_int_width: libloading::Symbol<CapabilityQuery> = library
            .get(b"cblas_inject_blas_int_width\0")
            .expect("missing cblas_inject_blas_int_width");
        let supports_lp64: libloading::Symbol<CapabilityQuery> = library
            .get(b"cblas_inject_supports_lp64_registration\0")
            .expect("missing cblas_inject_supports_lp64_registration");
        let supports_ilp64: libloading::Symbol<CapabilityQuery> = library
            .get(b"cblas_inject_supports_ilp64_registration\0")
            .expect("missing cblas_inject_supports_ilp64_registration");

        assert_eq!(blas_int_width(), expected_cblas_int_width());
        assert_eq!(supports_lp64(), 1);
        assert_eq!(supports_ilp64(), 1);

        let register_dgemm: libloading::Symbol<RegisterGemm> = library
            .get(register_dgemm_symbol())
            .expect("missing opposite-width dgemm registration symbol");
        let register_zgemm: libloading::Symbol<RegisterGemm> = library
            .get(register_zgemm_symbol())
            .expect("missing opposite-width zgemm registration symbol");
        let cblas_dgemm: libloading::Symbol<CblasDgemm> =
            library.get(b"cblas_dgemm\0").expect("missing cblas_dgemm");
        let cblas_zgemm: libloading::Symbol<CblasZgemm> =
            library.get(b"cblas_zgemm\0").expect("missing cblas_zgemm");
        let cblas_dgemm_64: libloading::Symbol<CblasDgemm64> = library
            .get(b"cblas_dgemm_64\0")
            .expect("missing cblas_dgemm_64");
        let cblas_zgemm_64: libloading::Symbol<CblasZgemm64> = library
            .get(b"cblas_zgemm_64\0")
            .expect("missing cblas_zgemm_64");

        assert_eq!(
            register_dgemm(mock_dgemm_opposite_width as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );
        assert_eq!(
            register_zgemm(mock_zgemm_opposite_width as *const c_void),
            CBLAS_INJECT_STATUS_OK
        );

        let a = [1.0; 12];
        let b = [2.0; 20];
        let mut c = [0.0; 6];
        cblas_dgemm(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
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

        assert_eq!(c[0], expected_sentinel());
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
        cblas_zgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
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

        assert_eq!(
            zc[0],
            Complex64::new(expected_sentinel(), -expected_sentinel())
        );
        assert_eq!(ZGEMM_CALLS.load(Ordering::SeqCst), 1);
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

        #[cfg(feature = "ilp64")]
        {
            let before_dgemm_calls = DGEMM_CALLS.load(Ordering::SeqCst);
            let mut fallback_c = [0.0; 6];
            cblas_dgemm_64(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                2,
                3,
                4,
                1.5,
                a.as_ptr(),
                2,
                b.as_ptr(),
                4,
                0.5,
                fallback_c.as_mut_ptr(),
                2,
            );
            assert_eq!(fallback_c[0], 32.0);
            assert_eq!(DGEMM_CALLS.load(Ordering::SeqCst), before_dgemm_calls + 1);
            assert_eq!(DGEMM_M.load(Ordering::SeqCst), 2);
            assert_eq!(DGEMM_N.load(Ordering::SeqCst), 3);
            assert_eq!(DGEMM_K.load(Ordering::SeqCst), 4);
            assert_eq!(DGEMM_LDA.load(Ordering::SeqCst), 2);
            assert_eq!(DGEMM_LDB.load(Ordering::SeqCst), 4);
            assert_eq!(DGEMM_LDC.load(Ordering::SeqCst), 2);

            let before_zgemm_calls = ZGEMM_CALLS.load(Ordering::SeqCst);
            let mut fallback_zc = [Complex64::new(0.0, 0.0); 6];
            cblas_zgemm_64(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                2,
                3,
                4,
                &alpha,
                za.as_ptr(),
                4,
                zb.as_ptr(),
                3,
                &beta,
                fallback_zc.as_mut_ptr(),
                3,
            );
            assert_eq!(fallback_zc[0], Complex64::new(32.0, -32.0));
            assert_eq!(ZGEMM_CALLS.load(Ordering::SeqCst), before_zgemm_calls + 1);
            assert_eq!(ZGEMM_M.load(Ordering::SeqCst), 3);
            assert_eq!(ZGEMM_N.load(Ordering::SeqCst), 2);
            assert_eq!(ZGEMM_K.load(Ordering::SeqCst), 4);
            assert_eq!(ZGEMM_LDA.load(Ordering::SeqCst), 3);
            assert_eq!(ZGEMM_LDB.load(Ordering::SeqCst), 4);
            assert_eq!(ZGEMM_LDC.load(Ordering::SeqCst), 3);

            let overflow = i64::from(i32::MAX) + 1;
            let before_dgemm_overflow_calls = DGEMM_CALLS.load(Ordering::SeqCst);
            let mut overflow_c = [11.0; 1];
            cblas_dgemm_64(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                overflow,
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
            assert_eq!(
                DGEMM_CALLS.load(Ordering::SeqCst),
                before_dgemm_overflow_calls
            );
            assert_eq!(overflow_c[0], 11.0);

            let before_zgemm_overflow_calls = ZGEMM_CALLS.load(Ordering::SeqCst);
            let mut overflow_zc = [Complex64::new(11.0, -11.0); 1];
            cblas_zgemm_64(
                CBLAS_COL_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                1,
                overflow,
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
            assert_eq!(
                ZGEMM_CALLS.load(Ordering::SeqCst),
                before_zgemm_overflow_calls
            );
            assert_eq!(overflow_zc[0], Complex64::new(11.0, -11.0));

            let register_dgemm_ilp64: libloading::Symbol<RegisterGemm> = library
                .get(b"cblas_inject_register_dgemm_ilp64\0")
                .expect("missing cblas_inject_register_dgemm_ilp64");
            let register_zgemm_ilp64: libloading::Symbol<RegisterGemm> = library
                .get(b"cblas_inject_register_zgemm_ilp64\0")
                .expect("missing cblas_inject_register_zgemm_ilp64");
            assert_eq!(
                register_dgemm_ilp64(mock_dgemm_ilp64_for_cblas64 as *const c_void),
                CBLAS_INJECT_STATUS_OK
            );
            assert_eq!(
                register_zgemm_ilp64(mock_zgemm_ilp64_for_cblas64 as *const c_void),
                CBLAS_INJECT_STATUS_OK
            );
        }

        let large_base = i64::from(i32::MAX) + 100;
        let before_dgemm_calls = DGEMM_CALLS.load(Ordering::SeqCst);
        let mut c64 = [0.0; 1];
        cblas_dgemm_64(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            large_base + 1,
            large_base + 2,
            large_base + 3,
            1.5,
            a.as_ptr(),
            large_base + 4,
            b.as_ptr(),
            large_base + 5,
            0.5,
            c64.as_mut_ptr(),
            large_base + 6,
        );
        #[cfg(feature = "ilp64")]
        assert_eq!(c64[0], 640.0);
        #[cfg(not(feature = "ilp64"))]
        assert_eq!(c64[0], 64.0);
        assert_eq!(DGEMM_CALLS.load(Ordering::SeqCst), before_dgemm_calls + 1);
        assert_eq!(DGEMM_M.load(Ordering::SeqCst), large_base + 1);
        assert_eq!(DGEMM_N.load(Ordering::SeqCst), large_base + 2);
        assert_eq!(DGEMM_K.load(Ordering::SeqCst), large_base + 3);
        assert_eq!(DGEMM_LDA.load(Ordering::SeqCst), large_base + 4);
        assert_eq!(DGEMM_LDB.load(Ordering::SeqCst), large_base + 5);
        assert_eq!(DGEMM_LDC.load(Ordering::SeqCst), large_base + 6);

        let before_zgemm_calls = ZGEMM_CALLS.load(Ordering::SeqCst);
        let mut zc64 = [Complex64::new(0.0, 0.0); 1];
        cblas_zgemm_64(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            large_base + 1,
            large_base + 2,
            large_base + 3,
            &alpha,
            za.as_ptr(),
            large_base + 4,
            zb.as_ptr(),
            large_base + 5,
            &beta,
            zc64.as_mut_ptr(),
            large_base + 6,
        );
        #[cfg(feature = "ilp64")]
        assert_eq!(zc64[0], Complex64::new(640.0, -640.0));
        #[cfg(not(feature = "ilp64"))]
        assert_eq!(zc64[0], Complex64::new(64.0, -64.0));
        assert_eq!(ZGEMM_CALLS.load(Ordering::SeqCst), before_zgemm_calls + 1);
        assert_eq!(ZGEMM_M.load(Ordering::SeqCst), large_base + 2);
        assert_eq!(ZGEMM_N.load(Ordering::SeqCst), large_base + 1);
        assert_eq!(ZGEMM_K.load(Ordering::SeqCst), large_base + 3);
        assert_eq!(ZGEMM_LDA.load(Ordering::SeqCst), large_base + 5);
        assert_eq!(ZGEMM_LDB.load(Ordering::SeqCst), large_base + 4);
        assert_eq!(ZGEMM_LDC.load(Ordering::SeqCst), large_base + 6);
    }
}

fn find_cdylib() -> PathBuf {
    let candidates = cdylib_candidates();
    first_existing_cdylib(&candidates).unwrap_or_else(|| {
        panic!(
            "could not find built libcblas_inject cdylib; run `cargo build` before \
             `cargo test --test dynamic_c_api`. Searched: {}",
            candidates
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

fn first_existing_cdylib(candidates: &[PathBuf]) -> Option<PathBuf> {
    candidates.iter().find(|path| path.is_file()).cloned()
}

#[test]
fn cdylib_selection_prefers_ordered_profile_candidates_over_newer_fallbacks() {
    let base =
        std::env::temp_dir().join(format!("cblas-inject-dynamic-c-api-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(base.join("profile")).expect("create profile temp dir");
    std::fs::create_dir_all(base.join("fallback")).expect("create fallback temp dir");

    let profile_candidate = base.join("profile").join(cdylib_file_name());
    let fallback_candidate = base.join("fallback").join(cdylib_file_name());
    std::fs::write(&profile_candidate, b"profile").expect("write profile candidate");
    std::thread::sleep(std::time::Duration::from_millis(10));
    std::fs::write(&fallback_candidate, b"fallback").expect("write fallback candidate");

    let selected = first_existing_cdylib(&[
        profile_candidate.clone(),
        base.join("missing").join(cdylib_file_name()),
        fallback_candidate,
    ])
    .expect("select existing cdylib");

    assert_eq!(selected, profile_candidate);
    let _ = std::fs::remove_dir_all(base);
}

fn cdylib_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(deps_dir) = exe.parent() {
            candidates.push(deps_dir.join(cdylib_file_name()));
        }
        if let Some(profile_dir) = exe.parent().and_then(|deps_dir| deps_dir.parent()) {
            candidates.push(profile_dir.join(cdylib_file_name()));
        }
    }

    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let target_dir = PathBuf::from(target_dir);
        candidates.push(target_dir.join("debug").join(cdylib_file_name()));
        candidates.push(target_dir.join("release").join(cdylib_file_name()));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(
        manifest_dir
            .join("target")
            .join("debug")
            .join(cdylib_file_name()),
    );
    candidates.push(
        manifest_dir
            .join("target")
            .join("release")
            .join(cdylib_file_name()),
    );
    candidates
}

fn cdylib_file_name() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "cblas_inject.dll"
    }
    #[cfg(target_os = "macos")]
    {
        "libcblas_inject.dylib"
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        "libcblas_inject.so"
    }
}

fn expected_cblas_int_width() -> c_int {
    #[cfg(feature = "ilp64")]
    {
        64
    }
    #[cfg(not(feature = "ilp64"))]
    {
        32
    }
}

fn expected_sentinel() -> f64 {
    #[cfg(feature = "ilp64")]
    {
        32.0
    }
    #[cfg(not(feature = "ilp64"))]
    {
        64.0
    }
}

fn register_dgemm_symbol() -> &'static [u8] {
    #[cfg(feature = "ilp64")]
    {
        b"cblas_inject_register_dgemm_lp64\0"
    }
    #[cfg(not(feature = "ilp64"))]
    {
        b"cblas_inject_register_dgemm_ilp64\0"
    }
}

fn register_zgemm_symbol() -> &'static [u8] {
    #[cfg(feature = "ilp64")]
    {
        b"cblas_inject_register_zgemm_lp64\0"
    }
    #[cfg(not(feature = "ilp64"))]
    {
        b"cblas_inject_register_zgemm_ilp64\0"
    }
}
