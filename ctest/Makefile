# Makefile for cblas-trampoline ctest
# Uses test files from extern/OpenBLAS/ctest/

OPENBLAS_CTEST = ../extern/OpenBLAS/ctest
VPATH = $(OPENBLAS_CTEST)

CC = cc
CFLAGS = -DADD_ -DCBLAS -I. -I$(OPENBLAS_CTEST)

# Link cblas-trampoline first (provides cblas_* symbols)
# Then OpenBLAS for Fortran BLAS (drot_, etc.) reference
TRAMPOLINE_LIB = ../target/release/libcblas_trampoline.dylib
LDFLAGS = -L../target/release -lcblas_trampoline -L/opt/homebrew/opt/openblas/lib -lopenblas

# BLAS Level 1 test objects
stestl1o = c_sblas1.o
dtestl1o = c_dblas1.o
ctestl1o = c_cblas1.o
ztestl1o = c_zblas1.o

# Test drivers
stestl1_driver = c_sblat1c.o
dtestl1_driver = c_dblat1c.o
ctestl1_driver = c_cblat1c.o
ztestl1_driver = c_zblat1c.o

.PHONY: all clean test test1 build-trampoline

all: build-trampoline xscblat1 xdcblat1 xccblat1 xzcblat1

build-trampoline:
	cd .. && cargo build --release --features openblas

# Compile sources
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# BLAS Level 1 test executables
xscblat1: $(stestl1_driver) $(stestl1o)
	$(CC) -o $@ $^ $(LDFLAGS)

xdcblat1: $(dtestl1_driver) $(dtestl1o)
	$(CC) -o $@ $^ $(LDFLAGS)

xccblat1: $(ctestl1_driver) $(ctestl1o)
	$(CC) -o $@ $^ $(LDFLAGS)

xzcblat1: $(ztestl1_driver) $(ztestl1o)
	$(CC) -o $@ $^ $(LDFLAGS)

test1: all
	DYLD_LIBRARY_PATH=../target/release ./xscblat1
	DYLD_LIBRARY_PATH=../target/release ./xdcblat1
	DYLD_LIBRARY_PATH=../target/release ./xccblat1
	DYLD_LIBRARY_PATH=../target/release ./xzcblat1

clean:
	rm -f xscblat1 xdcblat1 xccblat1 xzcblat1
	rm -f *.o
