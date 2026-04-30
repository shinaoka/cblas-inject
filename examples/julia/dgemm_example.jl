#!/usr/bin/env julia
#
# Example: Using cblas-inject from Julia
#
# This demonstrates how to:
# 1. Get Fortran BLAS function pointers from libblastrampoline
# 2. Register them with cblas-inject
# 3. Call CBLAS-style functions
#
# Prerequisites:
#   # Recommended default build:
#   cargo build --release
#
# Usage:
#   julia examples/julia/dgemm_example.jl

using LinearAlgebra
using Libdl

# Path to compiled cblas-inject library
const CBLAS_LIB = joinpath(@__DIR__, "../../target/release/libcblas_inject")

# CBLAS constants
const CblasColMajor = 102
const CblasNoTrans = 111

"""
Detect whether Julia's BLAS provider uses ILP64 (64-bit integers) or LP64 (32-bit integers).
"""
function detect_blas_interface()
    return LinearAlgebra.BLAS.USE_BLAS64 ? :ilp64 : :lp64
end

# Determine BLAS interface at load time
const BLAS_INTERFACE = detect_blas_interface()

function register_dgemm(lib, dgemm_ptr, interface)
    register_name = if interface == :ilp64
        :cblas_inject_register_dgemm_ilp64
    else
        :cblas_inject_register_dgemm_lp64
    end

    register_fn = dlsym(lib, register_name)
    status = ccall(register_fn, Cint, (Ptr{Cvoid},), dgemm_ptr)
    status == 0 || error("cblas-inject registration failed: $status")
end

function main()
    println("Detected BLAS interface: $BLAS_INTERFACE")

    # Load cblas-inject library
    lib = if Sys.isapple()
        dlopen("$(CBLAS_LIB).dylib")
    elseif Sys.islinux()
        dlopen("$(CBLAS_LIB).so")
    elseif Sys.iswindows()
        dlopen("$(CBLAS_LIB).dll")
    else
        error("Unsupported platform")
    end

    println("Loaded cblas-inject library")

    # Get Fortran BLAS function pointer from libblastrampoline
    dgemm_ptr = LinearAlgebra.BLAS.lbt_get_forward("dgemm_", BLAS_INTERFACE)
    println("Got dgemm_ pointer: $dgemm_ptr")

    if dgemm_ptr == C_NULL
        error("Failed to get dgemm_ pointer. Make sure BLAS is properly configured.")
    end

    # Register with the API matching Julia's BLAS provider ABI
    register_dgemm(lib, dgemm_ptr, BLAS_INTERFACE)
    println("Registered dgemm")

    # Get cblas_dgemm symbol
    cblas_dgemm = dlsym(lib, :cblas_dgemm)
    cblas_int_width = ccall(dlsym(lib, :cblas_inject_blas_int_width), Cint, ())

    # Matrix dimensions
    alpha = 1.0
    beta = 0.0

    # Column-major matrices (Julia's native layout)
    # A: m×k, B: k×n, C: m×n
    A = Float64[1 2; 3 4; 5 6]  # 3×2
    B = Float64[1 2 3 4; 5 6 7 8]  # 2×4
    C = zeros(Float64, 3, 4)

    # Call cblas_dgemm / cblas_dgemm_64 with column-major layout
    # For column-major: lda >= m, ldb >= k, ldc >= m
    if cblas_int_width == 64
        cblas_dgemm_64 = dlsym(lib, :cblas_dgemm_64)
        ccall(
            cblas_dgemm_64, Cvoid,
            (Cint, Cint, Cint,           # Order, TransA, TransB
             Int64, Int64, Int64,        # M, N, K
             Float64,                    # alpha
             Ptr{Float64}, Int64,        # A, lda
             Ptr{Float64}, Int64,        # B, ldb
             Float64,                    # beta
             Ptr{Float64}, Int64),       # C, ldc
            Cint(CblasColMajor), Cint(CblasNoTrans), Cint(CblasNoTrans),
            Int64(3), Int64(4), Int64(2),
            alpha,
            pointer(A), Int64(3),  # lda = m for column-major NoTrans
            pointer(B), Int64(2),  # ldb = k for column-major NoTrans
            beta,
            pointer(C), Int64(3)   # ldc = m for column-major
        )
    elseif cblas_int_width == 32
        ccall(
            cblas_dgemm, Cvoid,
            (Cint, Cint, Cint,           # Order, TransA, TransB
             Int32, Int32, Int32,        # M, N, K
             Float64,                    # alpha
             Ptr{Float64}, Int32,        # A, lda
             Ptr{Float64}, Int32,        # B, ldb
             Float64,                    # beta
             Ptr{Float64}, Int32),       # C, ldc
            Cint(CblasColMajor), Cint(CblasNoTrans), Cint(CblasNoTrans),
            Int32(3), Int32(4), Int32(2),
            alpha,
            pointer(A), Int32(3),  # lda = m for column-major NoTrans
            pointer(B), Int32(2),  # ldb = k for column-major NoTrans
            beta,
            pointer(C), Int32(3)   # ldc = m for column-major
        )
    else
        error("Unsupported cblas-inject integer width: $cblas_int_width")
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
