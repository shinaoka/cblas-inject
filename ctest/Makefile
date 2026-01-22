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

# BLAS Level 2 test objects
stestl2o = c_sblas2.o c_s2chke.o auxiliary.o c_xerbla.o constant.o
dtestl2o = c_dblas2.o c_d2chke.o auxiliary.o c_xerbla.o constant.o
ctestl2o = c_cblas2.o c_c2chke.o auxiliary.o c_xerbla.o constant.o
ztestl2o = c_zblas2.o c_z2chke.o auxiliary.o c_xerbla.o constant.o

# Test drivers
stestl1_driver = c_sblat1c.o
dtestl1_driver = c_dblat1c.o
ctestl1_driver = c_cblat1c.o
ztestl1_driver = c_zblat1c.o

stestl2_driver = c_sblat2c.o
dtestl2_driver = c_dblat2c.o
ctestl2_driver = c_cblat2c.o
ztestl2_driver = c_zblat2c.o

.PHONY: all clean test test1 test2 build-trampoline

all: build-trampoline xscblat1 xdcblat1 xccblat1 xzcblat1 xscblat2 xdcblat2 xccblat2 xzcblat2

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

# BLAS Level 2 test executables
xscblat2: $(stestl2_driver) $(stestl2o)
	$(CC) -o $@ $^ $(LDFLAGS)

xdcblat2: $(dtestl2_driver) $(dtestl2o)
	$(CC) -o $@ $^ $(LDFLAGS)

xccblat2: $(ctestl2_driver) $(ctestl2o)
	$(CC) -o $@ $^ $(LDFLAGS)

xzcblat2: $(ztestl2_driver) $(ztestl2o)
	$(CC) -o $@ $^ $(LDFLAGS)

test1: all
	./xscblat1
	./xdcblat1
	./xccblat1
	./xzcblat1

test2: all
	./xscblat2 < $(OPENBLAS_CTEST)/sin2
	./xdcblat2 < $(OPENBLAS_CTEST)/din2
	./xccblat2 < $(OPENBLAS_CTEST)/cin2
	./xzcblat2 < $(OPENBLAS_CTEST)/zin2

clean:
	rm -f xscblat1 xdcblat1 xccblat1 xzcblat1
	rm -f xscblat2 xdcblat2 xccblat2 xzcblat2
	rm -f *.o
