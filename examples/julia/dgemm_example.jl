#!/usr/bin/env julia
#
# Example: Using cblas-runtime from Julia
#
# This demonstrates how to:
# 1. Get Fortran BLAS function pointers from libblastrampoline
# 2. Register them with cblas-runtime
# 3. Call CBLAS-style functions
#
# Prerequisites:
#   # For LP64 (32-bit integers):
#   cargo build --release
#
#   # For ILP64 (64-bit integers, typical on modern Julia):
#   cargo build --release --features ilp64
#
# Usage:
#   julia examples/julia/dgemm_example.jl

using LinearAlgebra
using Libdl

# Path to compiled cblas-runtime library
const CBLAS_LIB = joinpath(@__DIR__, "../../target/release/libcblas_runtime")

# CBLAS constants
const CblasColMajor = 102
const CblasNoTrans = 111

"""
Detect whether the loaded BLAS uses ILP64 (64-bit integers) or LP64 (32-bit integers).
"""
function detect_blas_interface()
    config = LinearAlgebra.BLAS.lbt_get_config()
    # Check if any loaded library uses ILP64
    for lib in config.loaded_libs
        if lib.interface == :ilp64
            return :ilp64
        end
    end
    return :lp64
end

# Determine BLAS interface at load time
const BLAS_INTERFACE = detect_blas_interface()
const USE_ILP64 = BLAS_INTERFACE == :ilp64

function main()
    println("Detected BLAS interface: $BLAS_INTERFACE")

    # Load cblas-runtime library
    lib = if Sys.isapple()
        dlopen("$(CBLAS_LIB).dylib")
    elseif Sys.islinux()
        dlopen("$(CBLAS_LIB).so")
    elseif Sys.iswindows()
        dlopen("$(CBLAS_LIB).dll")
    else
        error("Unsupported platform")
    end

    println("Loaded cblas-runtime library")

    # Get Fortran BLAS function pointer from libblastrampoline
    dgemm_ptr = LinearAlgebra.BLAS.lbt_get_forward("dgemm_", BLAS_INTERFACE)
    println("Got dgemm_ pointer: $dgemm_ptr")

    if dgemm_ptr == C_NULL
        error("Failed to get dgemm_ pointer. Make sure BLAS is properly configured.")
    end

    # Register with cblas-runtime
    register_dgemm = dlsym(lib, :register_dgemm)
    ccall(register_dgemm, Cvoid, (Ptr{Cvoid},), dgemm_ptr)
    println("Registered dgemm")

    # Get cblas_dgemm symbol
    cblas_dgemm = dlsym(lib, :cblas_dgemm)

    # Matrix dimensions
    alpha = 1.0
    beta = 0.0

    # Column-major matrices (Julia's native layout)
    # A: m×k, B: k×n, C: m×n
    A = Float64[1 2; 3 4; 5 6]  # 3×2
    B = Float64[1 2 3 4; 5 6 7 8]  # 2×4
    C = zeros(Float64, 3, 4)

    # Call cblas_dgemm with column-major layout
    # For column-major: lda >= m, ldb >= k, ldc >= m
    if USE_ILP64
        ccall(
            cblas_dgemm, Cvoid,
            (Int64, Int64, Int64,        # Order, TransA, TransB
             Int64, Int64, Int64,        # M, N, K
             Float64,                    # alpha
             Ptr{Float64}, Int64,        # A, lda
             Ptr{Float64}, Int64,        # B, ldb
             Float64,                    # beta
             Ptr{Float64}, Int64),       # C, ldc
            Int64(CblasColMajor), Int64(CblasNoTrans), Int64(CblasNoTrans),
            Int64(3), Int64(4), Int64(2),
            alpha,
            pointer(A), Int64(3),  # lda = m for column-major NoTrans
            pointer(B), Int64(2),  # ldb = k for column-major NoTrans
            beta,
            pointer(C), Int64(3)   # ldc = m for column-major
        )
    else
        ccall(
            cblas_dgemm, Cvoid,
            (Int32, Int32, Int32,        # Order, TransA, TransB
             Int32, Int32, Int32,        # M, N, K
             Float64,                    # alpha
             Ptr{Float64}, Int32,        # A, lda
             Ptr{Float64}, Int32,        # B, ldb
             Float64,                    # beta
             Ptr{Float64}, Int32),       # C, ldc
            Int32(CblasColMajor), Int32(CblasNoTrans), Int32(CblasNoTrans),
            Int32(3), Int32(4), Int32(2),
            alpha,
            pointer(A), Int32(3),  # lda = m for column-major NoTrans
            pointer(B), Int32(2),  # ldb = k for column-major NoTrans
            beta,
            pointer(C), Int32(3)   # ldc = m for column-major
        )
    end

    println("\nResult C = A × B:")
    println("A = ")
    display(A)
    println("\nB = ")
    display(B)
    println("\nC = ")
    display(C)

    # Verify with Julia's built-in multiplication
    expected = A * B
    println("\nExpected (Julia A * B):")
    display(expected)

    if isapprox(C, expected)
        println("\n✓ Results match!")
    else
        println("\n✗ Results don't match!")
        exit(1)
    end

    dlclose(lib)
end

main()
