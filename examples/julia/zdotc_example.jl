#!/usr/bin/env julia
#
# Example: Complex dot product with return style configuration
#
# This demonstrates how to configure the complex return ABI for
# cdotc, cdotu, zdotc, zdotu functions.
#
# Fortran complex functions have two calling conventions:
# - ReturnValue (0): Complex returned via register (OpenBLAS, MKL intel, BLIS)
# - HiddenArgument (1): Complex written to first pointer arg (gfortran default, MKL gf)
#
# Prerequisites:
#   cargo build --release --features ilp64
#
# Usage:
#   julia examples/julia/zdotc_example.jl

using LinearAlgebra
using Libdl

# Path to compiled cblas-inject library
const CBLAS_LIB = joinpath(@__DIR__, "../../target/release/libcblas_inject")

# Complex return style constants (must match Rust enum)
const ComplexReturnStyle_ReturnValue = Int32(0)
const ComplexReturnStyle_HiddenArgument = Int32(1)

# CBLAS constants
const CblasColMajor = 102

"""
Detect whether the loaded BLAS uses ILP64 (64-bit integers) or LP64 (32-bit integers).
"""
function detect_blas_interface()
    config = LinearAlgebra.BLAS.lbt_get_config()
    for lib in config.loaded_libs
        if lib.interface == :ilp64
            return :ilp64
        end
    end
    return :lp64
end

function main()
    # Detect BLAS interface
    interface = detect_blas_interface()
    println("Detected BLAS interface: $interface")

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

    # Get function pointers from libblastrampoline
    zdotc_ptr = LinearAlgebra.BLAS.lbt_get_forward("zdotc_", interface)
    println("Got zdotc_ pointer: $zdotc_ptr")

    if zdotc_ptr == C_NULL
        error("Failed to get zdotc_ pointer. Make sure BLAS is properly configured.")
    end

    # Set the complex return style BEFORE registering
    # OpenBLAS uses ReturnValue convention
    set_complex_return_style = dlsym(lib, :set_complex_return_style)
    ccall(set_complex_return_style, Cvoid, (Int32,), ComplexReturnStyle_ReturnValue)
    println("Set complex return style to ReturnValue")

    # Now register zdotc
    register_zdotc = dlsym(lib, :register_zdotc)
    ccall(register_zdotc, Cvoid, (Ptr{Cvoid},), zdotc_ptr)
    println("Registered zdotc")

    # Get cblas_zdotc_sub
    cblas_zdotc_sub = dlsym(lib, :cblas_zdotc_sub)

    # Test vectors
    n = 4
    x = ComplexF64[1+2im, 3+4im, 5+6im, 7+8im]
    y = ComplexF64[1-1im, 2-2im, 3-3im, 4-4im]
    result = Ref{ComplexF64}(0.0 + 0.0im)

    # Call cblas_zdotc_sub
    # zdotc computes: conj(x)' * y = sum(conj(x[i]) * y[i])
    if interface == :ilp64
        ccall(
            cblas_zdotc_sub, Cvoid,
            (Int64, Ptr{ComplexF64}, Int64, Ptr{ComplexF64}, Int64, Ptr{ComplexF64}),
            Int64(n), pointer(x), Int64(1), pointer(y), Int64(1), result
        )
    else
        ccall(
            cblas_zdotc_sub, Cvoid,
            (Int32, Ptr{ComplexF64}, Int32, Ptr{ComplexF64}, Int32, Ptr{ComplexF64}),
            Int32(n), pointer(x), Int32(1), pointer(y), Int32(1), result
        )
    end

    println("\nzdotc(x, y) where:")
    println("  x = $x")
    println("  y = $y")
    println("  result = $(result[])")

    # Verify with Julia
    # zdotc = conj(x)' * y = dot(x, y) in Julia
    expected = dot(x, y)
    println("  expected (Julia dot(x,y)) = $expected")

    if isapprox(result[], expected)
        println("\n✓ Results match!")
    else
        println("\n✗ Results don't match!")
        exit(1)
    end

    dlclose(lib)
end

main()
