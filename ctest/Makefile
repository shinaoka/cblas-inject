# Makefile for cblas-trampoline ctest
#
# This builds the OpenBLAS-derived CBLAS test suite against cblas-trampoline.
#
# Usage:
#   make           # Build LP64 test binaries
#   make test      # Run LP64 tests
#   make ILP64=1   # Build ILP64 test binaries
#   make clean     # Remove built files

CC ?= cc
CFLAGS = -O2 -Wall

# Fortran symbol naming convention (trailing underscore)
CFLAGS += -DADD_

# ILP64 support
ifdef ILP64
CFLAGS += -DUSE64BITINT
SUFFIX = _64
else
SUFFIX =
endif

# Include paths
CFLAGS += -I.

# Library paths
# cblas-trampoline is linked dynamically or we use the Rust test harness
# For standalone testing, link against OpenBLAS for the Fortran BLAS backend
LDFLAGS = -L/usr/local/lib -lopenblas -lm

# Source files
COMMON_SRCS = auxiliary.c c_xerbla.c constant.c
DBLAT3_SRCS = c_dblat3c.c c_dblas3.c c_d3chke.c

# Object files
COMMON_OBJS = $(COMMON_SRCS:.c=.o)
DBLAT3_OBJS = $(DBLAT3_SRCS:.c=.o)

# Targets
.PHONY: all clean test test-dblat3

all: xdcblat3$(SUFFIX)

xdcblat3$(SUFFIX): $(DBLAT3_OBJS) $(COMMON_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Pattern rule for object files
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Test targets
test: test-dblat3

test-dblat3: xdcblat3$(SUFFIX)
	./xdcblat3$(SUFFIX) < din3

clean:
	rm -f *.o xdcblat3 xdcblat3_64

# Dependencies
auxiliary.o: common.h cblas_test.h cblas.h
c_xerbla.o: common.h cblas_test.h cblas.h
constant.o: cblas_test.h cblas.h
c_dblat3c.o: common.h
c_dblas3.o: common.h cblas_test.h cblas.h
c_d3chke.o: common.h cblas_test.h cblas.h
