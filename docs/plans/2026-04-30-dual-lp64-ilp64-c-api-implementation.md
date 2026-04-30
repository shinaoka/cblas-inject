# Dual LP64/ILP64 C API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an explicit dual LP64/ILP64 C registration ABI while preserving `cblas-sys` LP64 compatibility.

**Architecture:** Keep unprefixed `cblas_*` symbols LP64-compatible. Store LP64 and ILP64 GEMM provider pointers separately, expose prefixed C registration APIs for both widths, and dispatch LP64 CBLAS calls to either an LP64 provider directly or an ILP64 provider through widening. Add true `cblas_*_64` ILP64 entry points after the registration path is covered.

**Tech Stack:** Rust 2021, `OnceLock`, `#[no_mangle] extern "C"`, `num_complex`, GitHub Actions, C dynamic loading smoke tests through `dlopen`/`dlsym`.

---

### Task 1: Add ILP64 CI Check

**Files:**
- Modify: `.github/workflows/ci.yml`

**Step 1: Add the workflow step**

In the `rust-tests` job, after `Install OpenBLAS` and before `cblas-sys compatibility test`, add:

```yaml
      - name: Check ILP64 build
        run: cargo check --features ilp64 --all-targets
        env:
          LIBRARY_PATH: /opt/homebrew/opt/openblas/lib:/usr/local/opt/openblas/lib
```

**Step 2: Verify locally**

Run:

```bash
cargo check --features ilp64 --all-targets
```

Expected: exit 0. Bench warnings are acceptable if unchanged from baseline.

**Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: check ILP64 build"
```

### Task 2: Add LP64/ILP64 GEMM Function Pointer Types

**Files:**
- Modify: `src/backend.rs`

**Step 1: Add explicit integer aliases near the top of `src/backend.rs`**

Add after imports:

```rust
type BlasInt32 = i32;
type BlasInt64 = i64;
```

**Step 2: Add explicit GEMM pointer types near current `DgemmFnPtr` and `ZgemmFnPtr`**

Add types that do not depend on the `ilp64` Cargo feature:

```rust
pub type DgemmLp64FnPtr = unsafe extern "C" fn(
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
);

pub type DgemmIlp64FnPtr = unsafe extern "C" fn(
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
);

pub type ZgemmLp64FnPtr = unsafe extern "C" fn(
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
);

pub type ZgemmIlp64FnPtr = unsafe extern "C" fn(
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
);
```

**Step 3: Keep legacy types**

Do not remove existing `DgemmFnPtr` or `ZgemmFnPtr`. They are still used by the current crate API and can be wired to LP64 in later tasks.

**Step 4: Verify**

Run:

```bash
cargo check --lib
```

Expected: exit 0.

**Step 5: Commit**

```bash
git add src/backend.rs
git commit -m "feat: add explicit GEMM ABI pointer types"
```

### Task 3: Add Stable C Registration API and Capability Queries

**Files:**
- Modify: `src/backend.rs`

**Step 1: Add status constants**

Near the new pointer type aliases or before registration functions, add:

```rust
const CBLAS_INJECT_STATUS_OK: i32 = 0;
const CBLAS_INJECT_STATUS_NULL_POINTER: i32 = 1;
const CBLAS_INJECT_STATUS_ALREADY_REGISTERED: i32 = 2;
```

**Step 2: Add separate storage**

Near current `static DGEMM`, `static ZGEMM`, add:

```rust
static DGEMM_LP64: OnceLock<DgemmLp64FnPtr> = OnceLock::new();
static DGEMM_ILP64: OnceLock<DgemmIlp64FnPtr> = OnceLock::new();
static ZGEMM_LP64: OnceLock<ZgemmLp64FnPtr> = OnceLock::new();
static ZGEMM_ILP64: OnceLock<ZgemmIlp64FnPtr> = OnceLock::new();
```

**Step 3: Add raw pointer registration functions**

Add after existing `register_dgemm`/`register_zgemm`:

```rust
#[no_mangle]
pub extern "C" fn cblas_inject_register_dgemm_lp64(f: *const std::ffi::c_void) -> i32 {
    if f.is_null() {
        return CBLAS_INJECT_STATUS_NULL_POINTER;
    }
    let f: DgemmLp64FnPtr = unsafe { std::mem::transmute(f) };
    match DGEMM_LP64.set(f) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}

#[no_mangle]
pub extern "C" fn cblas_inject_register_dgemm_ilp64(f: *const std::ffi::c_void) -> i32 {
    if f.is_null() {
        return CBLAS_INJECT_STATUS_NULL_POINTER;
    }
    let f: DgemmIlp64FnPtr = unsafe { std::mem::transmute(f) };
    match DGEMM_ILP64.set(f) {
        Ok(()) => CBLAS_INJECT_STATUS_OK,
        Err(_) => CBLAS_INJECT_STATUS_ALREADY_REGISTERED,
    }
}
```

Repeat the same pattern for `zgemm`.

**Step 4: Make legacy registration fill LP64 storage**

Update legacy `register_dgemm` and `register_zgemm` so they keep old behavior but also populate the LP64 storage when the build ABI is LP64:

```rust
#[cfg(not(feature = "ilp64"))]
{
    let _ = DGEMM_LP64.set(std::mem::transmute::<DgemmFnPtr, DgemmLp64FnPtr>(f));
}
```

If this transmute is rejected by the compiler under feature combinations, replace it with an explicit LP64-only helper:

```rust
#[cfg(not(feature = "ilp64"))]
unsafe fn legacy_dgemm_to_lp64(f: DgemmFnPtr) -> DgemmLp64FnPtr {
    std::mem::transmute(f)
}
```

**Step 5: Add query functions**

Add:

```rust
#[no_mangle]
pub extern "C" fn cblas_inject_blas_int_width() -> i32 {
    32
}

#[no_mangle]
pub extern "C" fn cblas_inject_supports_lp64_registration() -> i32 {
    1
}

#[no_mangle]
pub extern "C" fn cblas_inject_supports_ilp64_registration() -> i32 {
    1
}
```

**Step 6: Verify**

Run:

```bash
cargo check --lib
cargo check --features ilp64 --lib
```

Expected: both exit 0.

**Step 7: Commit**

```bash
git add src/backend.rs
git commit -m "feat: expose dual-width GEMM registration ABI"
```

### Task 4: Add Backend Dispatch Helpers

**Files:**
- Modify: `src/backend.rs`
- Modify: `src/blas3/gemm.rs`

**Step 1: Add provider enum in `src/backend.rs`**

Add near GEMM getters:

```rust
pub(crate) enum DgemmProvider {
    Lp64(DgemmLp64FnPtr),
    Ilp64(DgemmIlp64FnPtr),
}

pub(crate) enum ZgemmProvider {
    Lp64(ZgemmLp64FnPtr),
    Ilp64(ZgemmIlp64FnPtr),
}
```

**Step 2: Add getters that prefer LP64 for LP64 CBLAS**

```rust
pub(crate) fn get_dgemm_for_lp64_cblas() -> DgemmProvider {
    if let Some(f) = DGEMM_LP64.get() {
        return DgemmProvider::Lp64(*f);
    }
    if let Some(f) = DGEMM_ILP64.get() {
        return DgemmProvider::Ilp64(*f);
    }
    panic!("dgemm not registered: call cblas_inject_register_dgemm_lp64() or cblas_inject_register_dgemm_ilp64() first");
}

pub(crate) fn get_zgemm_for_lp64_cblas() -> ZgemmProvider {
    if let Some(f) = ZGEMM_LP64.get() {
        return ZgemmProvider::Lp64(*f);
    }
    if let Some(f) = ZGEMM_ILP64.get() {
        return ZgemmProvider::Ilp64(*f);
    }
    panic!("zgemm not registered: call cblas_inject_register_zgemm_lp64() or cblas_inject_register_zgemm_ilp64() first");
}
```

**Step 3: Add local call helpers in `src/blas3/gemm.rs`**

Import the new enums:

```rust
use crate::backend::{DgemmProvider, ZgemmProvider, get_dgemm_for_lp64_cblas, get_zgemm_for_lp64_cblas};
```

Add helpers:

```rust
#[allow(clippy::too_many_arguments)]
unsafe fn call_dgemm_provider(
    provider: DgemmProvider,
    transa: *const std::ffi::c_char,
    transb: *const std::ffi::c_char,
    m: blasint,
    n: blasint,
    k: blasint,
    alpha: *const f64,
    a: *const f64,
    lda: blasint,
    b: *const f64,
    ldb: blasint,
    beta: *const f64,
    c: *mut f64,
    ldc: blasint,
) {
    match provider {
        DgemmProvider::Lp64(f) => {
            let m32 = m as i32;
            let n32 = n as i32;
            let k32 = k as i32;
            let lda32 = lda as i32;
            let ldb32 = ldb as i32;
            let ldc32 = ldc as i32;
            f(transa, transb, &m32, &n32, &k32, alpha, a, &lda32, b, &ldb32, beta, c, &ldc32);
        }
        DgemmProvider::Ilp64(f) => {
            let m64 = m as i64;
            let n64 = n as i64;
            let k64 = k as i64;
            let lda64 = lda as i64;
            let ldb64 = ldb as i64;
            let ldc64 = ldc as i64;
            f(transa, transb, &m64, &n64, &k64, alpha, a, &lda64, b, &ldb64, beta, c, &ldc64);
        }
    }
}
```

Add equivalent `call_zgemm_provider`.

**Step 4: Update `cblas_dgemm` and `cblas_zgemm`**

Replace `let dgemm = get_dgemm();` with:

```rust
let dgemm = get_dgemm_for_lp64_cblas();
```

Replace direct `dgemm(...)` calls with `call_dgemm_provider(dgemm, ...)`.

Because `DgemmProvider` is moved, derive `Copy, Clone` on provider enums or call the getter inside each match branch. Prefer `#[derive(Clone, Copy)]`.

Repeat for `zgemm`.

**Step 5: Verify**

Run:

```bash
cargo test --test gemm
cargo check --features ilp64 --lib
```

Expected: both exit 0.

**Step 6: Commit**

```bash
git add src/backend.rs src/blas3/gemm.rs
git commit -m "feat: dispatch GEMM through dual-width providers"
```

### Task 5: Add Dynamic-Library Smoke Tests

**Files:**
- Modify: `Cargo.toml`
- Create: `tests/dynamic_c_api.rs`

**Step 1: Add dev-dependency**

In `Cargo.toml`:

```toml
[dev-dependencies]
libloading = "0.8"
```

Keep existing dev-dependencies.

**Step 2: Write failing smoke test**

Create `tests/dynamic_c_api.rs` with tests that load the current test dynamic library and call registration symbols. Use `std::env::var("CARGO_TARGET_DIR").unwrap_or_else(...)` fallback to `target`, and platform-specific library names:

```rust
#[cfg(target_os = "linux")]
const LIB_NAME: &str = "libcblas_inject.so";
#[cfg(target_os = "macos")]
const LIB_NAME: &str = "libcblas_inject.dylib";
#[cfg(target_os = "windows")]
const LIB_NAME: &str = "cblas_inject.dll";
```

Define mock ILP64 providers:

```rust
static LAST_DGEMM_M: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);

unsafe extern "C" fn mock_dgemm_ilp64(
    _transa: *const std::ffi::c_char,
    _transb: *const std::ffi::c_char,
    m: *const i64,
    _n: *const i64,
    _k: *const i64,
    _alpha: *const f64,
    _a: *const f64,
    _lda: *const i64,
    _b: *const f64,
    _ldb: *const i64,
    _beta: *const f64,
    c: *mut f64,
    _ldc: *const i64,
) {
    LAST_DGEMM_M.store(*m, std::sync::atomic::Ordering::SeqCst);
    *c = 42.0;
}
```

Test outline:

```rust
#[test]
fn dynamic_api_registers_ilp64_dgemm_and_lp64_cblas_dispatches_to_it() {
    // load cdylib
    // dlsym cblas_inject_register_dgemm_ilp64
    // dlsym cblas_dgemm
    // register mock_dgemm_ilp64
    // call cblas_dgemm with LP64 m=3
    // assert registration returned 0, C[0] == 42.0, LAST_DGEMM_M == 3
}
```

Add a matching `zgemm` test that records widened dimensions and writes a complex sentinel.

**Step 3: Run test and verify RED**

Run:

```bash
cargo build --release
cargo test --test dynamic_c_api -- --nocapture
```

Expected before implementation tasks are complete: failure from missing symbols or dispatch not reaching ILP64 provider. If this is run after previous tasks, it should pass; document that in the PR.

**Step 4: Verify GREEN**

Run:

```bash
cargo build --release
cargo test --test dynamic_c_api -- --nocapture
```

Expected after Task 4: exit 0.

**Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock tests/dynamic_c_api.rs
git commit -m "test: cover dynamic dual-width GEMM registration"
```

### Task 6: Add True ILP64 CBLAS GEMM Symbols

**Files:**
- Modify: `src/backend.rs`
- Modify: `src/blas3/gemm.rs`
- Modify: `tests/dynamic_c_api.rs`

**Step 1: Add ILP64-preferring getters**

In `src/backend.rs`:

```rust
pub(crate) fn get_dgemm_for_ilp64_cblas() -> DgemmProvider {
    if let Some(f) = DGEMM_ILP64.get() {
        return DgemmProvider::Ilp64(*f);
    }
    if let Some(f) = DGEMM_LP64.get() {
        return DgemmProvider::Lp64(*f);
    }
    panic!("dgemm not registered");
}
```

Repeat for `zgemm`.

**Step 2: Add narrowing helper**

In `src/blas3/gemm.rs`:

```rust
fn narrow_blasint(value: i64, name: &str) -> i32 {
    i32::try_from(value).unwrap_or_else(|_| panic!("{name} does not fit LP64 BLAS integer"))
}
```

Use this only when `cblas_*_64` must call an LP64 provider.

**Step 3: Add `cblas_dgemm_64`**

Add a new `#[no_mangle] pub unsafe extern "C" fn cblas_dgemm_64(...)` with the same argument list as `cblas_dgemm`, but all BLAS integer arguments are `i64`. It should:

- Preserve the current row-major conversion logic.
- Prefer `get_dgemm_for_ilp64_cblas()`.
- Call ILP64 providers directly.
- Narrow and call LP64 providers only if all integer arguments fit `i32`.

**Step 4: Add `cblas_zgemm_64`**

Mirror `cblas_dgemm_64` for complex double GEMM.

**Step 5: Extend dynamic tests**

Add tests that resolve `cblas_dgemm_64` and `cblas_zgemm_64`, register ILP64 providers, and verify dimensions are passed as `i64` without narrowing.

**Step 6: Verify**

Run:

```bash
cargo test --test dynamic_c_api -- --nocapture
cargo test --test gemm
cargo check --features ilp64 --all-targets
```

Expected: all exit 0.

**Step 7: Commit**

```bash
git add src/backend.rs src/blas3/gemm.rs tests/dynamic_c_api.rs
git commit -m "feat: add ILP64 GEMM CBLAS symbols"
```

### Task 7: Add Generated or Handwritten C Header

**Files:**
- Create: `include/cblas_inject.h`
- Modify: `Cargo.toml` only if packaging metadata should include the header

**Step 1: Create header**

Create `include/cblas_inject.h` with:

```c
#ifndef CBLAS_INJECT_H
#define CBLAS_INJECT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int cblas_inject_blas_int_width(void);
int cblas_inject_supports_lp64_registration(void);
int cblas_inject_supports_ilp64_registration(void);

int cblas_inject_register_dgemm_lp64(const void *dgemm);
int cblas_inject_register_dgemm_ilp64(const void *dgemm64);
int cblas_inject_register_zgemm_lp64(const void *zgemm);
int cblas_inject_register_zgemm_ilp64(const void *zgemm64);

#ifdef __cplusplus
}
#endif

#endif
```

**Step 2: Add signature comments**

Document exact LP64 and ILP64 Fortran provider signatures for `dgemm` and `zgemm` in comments. Use `int32_t *` for LP64 and `int64_t *` for ILP64 dimensions and leading dimensions.

**Step 3: Verify**

Run:

```bash
cargo test --test dynamic_c_api
```

Expected: exit 0. Header creation does not affect Rust compilation directly.

**Step 4: Commit**

```bash
git add include/cblas_inject.h
git commit -m "docs: add cblas-inject C header"
```

### Task 8: Update README Last

**Files:**
- Modify: `README.md`

**Step 1: Update compatibility wording**

Change the feature list so it says:

```markdown
- **cblas-sys compatible**: Unprefixed `cblas_*` symbols use the LP64 CBLAS ABI expected by `cblas-sys`.
- **Dual provider registration**: Hosts can register LP64 or ILP64 Fortran BLAS provider pointers at runtime.
```

**Step 2: Update Julia example**

Replace the current ILP64-only example with runtime selection:

```julia
interface = LinearAlgebra.BLAS.USE_BLAS64 ? :ilp64 : :lp64
dgemm_ptr = LinearAlgebra.BLAS.lbt_get_forward("dgemm_", interface)
register_name = LinearAlgebra.BLAS.USE_BLAS64 ?
    :cblas_inject_register_dgemm_ilp64 :
    :cblas_inject_register_dgemm_lp64
register_dgemm = dlsym(lib, register_name)
status = ccall(register_dgemm, Cint, (Ptr{Cvoid},), dgemm_ptr)
status == 0 || error("Failed to register dgemm provider: status=$status")
```

**Step 3: Add C ABI section**

Document:

- `cblas_inject_blas_int_width()`
- `cblas_inject_register_dgemm_lp64`
- `cblas_inject_register_dgemm_ilp64`
- `cblas_inject_register_zgemm_lp64`
- `cblas_inject_register_zgemm_ilp64`
- `cblas_dgemm_64` / `cblas_zgemm_64` if Task 6 is implemented

**Step 4: Add same-instance warning**

Add a short note:

```markdown
Registration and CBLAS calls must use the same loaded `libcblas_inject` instance. If a downstream shared library links one copy while the host `dlopen`s another path, the provider registry will not be shared.
```

**Step 5: Verify**

Run:

```bash
cargo test --test dynamic_c_api
cargo test --test cblas_sys_compat
cargo check --features ilp64 --all-targets
```

Expected: all exit 0.

**Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document dual LP64 ILP64 registration"
```
