# LP64/ILP64 C API Design

## Context

`cblas-inject` is intended to serve as a CBLAS-compatible symbol provider for
Rust crates that already call CBLAS through crates such as `cblas-sys`, while a
host language such as Julia registers runtime BLAS provider pointers into the
same loaded library instance.

`cblas-sys` uses `c_int` for CBLAS dimensions and leading dimensions, and does
not expose an ILP64 variant. Therefore the existing `cblas_*` symbols must
remain LP64-compatible if `cblas-inject` is to stay usable as a drop-in provider
for `cblas-sys` callers.

Julia can determine the active BLAS interface at runtime through
`LinearAlgebra.BLAS.USE_BLAS64` and can obtain provider pointers through
`LinearAlgebra.BLAS.lbt_get_forward(name, :lp64 | :ilp64)`. The Rust/C library
cannot assume which interface will be used until the host process initializes
it.

## Goals

- Keep `cblas_*` ABI-compatible with `cblas-sys` LP64 callers.
- Expose stable C registration functions for both LP64 and ILP64 Fortran BLAS
  provider pointers.
- Allow a Julia ILP64 BLAS provider to back existing LP64 `cblas-sys` callers by
  widening dimensions internally.
- Provide a path to true ILP64 CBLAS entry points without breaking existing
  LP64 symbols.
- Let hosts query the loaded `cblas-inject` ABI capabilities before
  registration.

## Non-Goals

- Do not make a single `cblas_dgemm` symbol accept both LP64 and ILP64 C
  signatures. That is not a stable C ABI because the integer argument widths are
  part of the function signature.
- Do not require downstream `cblas-sys` users to change their Rust code to gain
  Julia ILP64 provider support.
- Do not remove legacy `register_*` symbols in the first pass.

## Public C ABI

Add prefixed registration functions that accept raw function pointers and return
status codes instead of panicking across FFI:

```c
int cblas_inject_register_dgemm_lp64(const void *dgemm);
int cblas_inject_register_dgemm_ilp64(const void *dgemm64);

int cblas_inject_register_zgemm_lp64(const void *zgemm);
int cblas_inject_register_zgemm_ilp64(const void *zgemm64);
```

The first implementation should cover `dgemm` and `zgemm`, then the pattern can
be generated or repeated for the rest of the supported BLAS surface.

Also expose capability queries:

```c
int cblas_inject_blas_int_width(void);
int cblas_inject_supports_lp64_registration(void);
int cblas_inject_supports_ilp64_registration(void);
```

`cblas_inject_blas_int_width()` reports the width of the legacy unprefixed
`cblas_*` ABI. With this design it should report `32` for the stable
`cblas-sys`-compatible ABI. ILP64 support is advertised separately because it is
a provider-registration and optional `*_64` symbol capability, not a change to
the legacy `cblas_*` signatures.

## Symbol Strategy

Keep these existing symbols LP64:

```c
cblas_dgemm(... int m, int n, int k, ... int lda, ...);
cblas_zgemm(... int m, int n, int k, ... int lda, ...);
```

Add true ILP64 CBLAS symbols only as separate names:

```c
cblas_dgemm_64(... int64_t m, int64_t n, int64_t k, ... int64_t lda, ...);
cblas_zgemm_64(... int64_t m, int64_t n, int64_t k, ... int64_t lda, ...);
```

The `*_64` symbols can be phased in after registration support. They are useful
for non-`cblas-sys` callers that want a direct ILP64 CBLAS ABI.

## Internal Dispatch

Store LP64 and ILP64 provider pointers separately:

```rust
static DGEMM_LP64: OnceLock<DgemmLp64FnPtr>;
static DGEMM_ILP64: OnceLock<DgemmIlp64FnPtr>;
```

Dispatch rules:

- `cblas_dgemm` receives LP64 arguments.
- If an LP64 provider is registered, call it directly.
- Otherwise, if an ILP64 provider is registered, widen all BLAS integer
  arguments from `i32` to `i64` and call the ILP64 provider.
- `cblas_dgemm_64` receives ILP64 arguments.
- If an ILP64 provider is registered, call it directly.
- Otherwise, if an LP64 provider is registered, range-check all BLAS integer
  arguments and narrow to `i32`; return an error path or call `cblas_xerbla` if
  values do not fit.
- If both providers are registered, each CBLAS entry point prefers the matching
  provider width.

The same rules apply to `zgemm`. Complex scalar arguments remain pointers for
CBLAS and Fortran complex GEMM provider signatures.

## Legacy Compatibility

Keep existing `register_dgemm` and `register_zgemm` as compatibility aliases for
LP64 registration. New host-language integrations should use the prefixed C ABI
because it is explicit about integer width and can return an error code.

The current `ilp64` Cargo feature, which changes the Rust `blasint` alias and
therefore changes the ABI of `cblas_*`, should no longer be the recommended
stable cdylib path. It may remain temporarily for Rust-internal compatibility
while the new explicit ABI is introduced.

## Julia Initialization Flow

Julia host code should:

1. Load the exact `libcblas_inject` instance used by downstream Rust shared
   libraries.
2. Check `LinearAlgebra.BLAS.USE_BLAS64`.
3. Resolve provider pointers with `lbt_get_forward("dgemm_", :ilp64)` or
   `lbt_get_forward("dgemm_", :lp64)`.
4. Call the matching `cblas_inject_register_*_{ilp64,lp64}` functions.
5. Let downstream Rust code call `cblas-sys` normally.

This preserves the `cblas-sys` call path while allowing Julia's ILP64 BLAS to be
the actual provider.

## Error Handling

New C ABI functions return integer status codes:

- `0`: success
- `1`: null function pointer
- `2`: already registered
- `3`: unsupported ABI width

Rust registration code must not panic across FFI. The existing unprefixed
registration functions may keep their current behavior initially, but the
prefixed API should be the documented stable interface.

## Testing

Add dynamic-library smoke tests that:

- Load the built `libcblas_inject` with `dlopen`.
- Resolve prefixed LP64 and ILP64 registration symbols with `dlsym`.
- Register mock `dgemm` and `zgemm` providers for LP64 and ILP64.
- Call `cblas_dgemm` and `cblas_zgemm` through the same loaded library.
- Verify that LP64 callers dispatch correctly to ILP64 providers by observing
  widened integer arguments in the mock provider.

CI should add at least:

```bash
cargo check --features ilp64 --all-targets
```

This is a guard for the current feature-gated implementation while the explicit
dual-ABI C surface is being introduced.

## Documentation

Update `README.md` at the end of the implementation so it reflects the final
API rather than an intermediate state. The README update should cover:

- `cblas-sys` compatibility means the unprefixed `cblas_*` symbols use the LP64
  CBLAS ABI.
- Julia hosts should choose LP64 or ILP64 registration based on
  `LinearAlgebra.BLAS.USE_BLAS64`.
- The stable C registration API is the prefixed
  `cblas_inject_register_*_{lp64,ilp64}` surface.
- Existing `register_*` symbols are legacy LP64 aliases.
- True ILP64 CBLAS calls, if implemented, use separate `cblas_*_64` symbols.
- Hosts must load/register against the same `libcblas_inject` instance that
  downstream Rust shared libraries use.

## Open Questions

- Should `cblas_*_64` be implemented in the first PR, or should the first PR
  focus only on dual-width provider registration behind LP64 `cblas-sys`
  symbols?
- Should the legacy `ilp64` feature eventually be deprecated for cdylib builds?
- Should registration for all BLAS functions be generated from a shared function
  table to avoid hand-maintaining LP64 and ILP64 variants?
