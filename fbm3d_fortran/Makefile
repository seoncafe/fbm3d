ifeq ($(origin FC), default)
FC = gfortran
endif

USE_OPENMP ?= 1

FC_NAME := $(notdir $(firstword $(FC)))

ifneq ($(filter ifort ifx,$(FC_NAME)),)
BASE_FFLAGS = -O3 -xHost -fp-model fast=2 -fpp
OMP_FLAG = -qopenmp
else
BASE_FFLAGS = -O3 -march=native -ffast-math -cpp
OMP_FLAG = -fopenmp
endif

FFTW_LIBS ?= $(shell pkg-config --libs fftw3f 2>/dev/null || echo -lfftw3f)
CFITSIO_LIBS ?= $(shell pkg-config --libs cfitsio 2>/dev/null || echo -lcfitsio)
HDF5_FFLAGS ?= $(shell pkg-config --cflags hdf5 2>/dev/null)
HDF5_LIBS ?= -lhdf5_fortran $(shell pkg-config --libs hdf5 2>/dev/null || echo -lhdf5)

ifeq ($(USE_OPENMP),1)
FFLAGS ?= $(BASE_FFLAGS) $(OMP_FLAG)
FFTW_THREAD_LIB ?= -lfftw3f_omp
else
FFLAGS ?= $(BASE_FFLAGS)
FFTW_THREAD_LIB =
endif

LDLIBS = $(FFTW_THREAD_LIB) $(FFTW_LIBS) $(CFITSIO_LIBS) $(HDF5_LIBS)

OBJS = \
	define.o \
	random_light.o \
	write_output.o \
	fbm3d.o

.PHONY: all clean cleanall debug

all: make_fbm3d.x

make_fbm3d.x: $(OBJS) make_fbm3d.o
	$(FC) $(FFLAGS) $^ $(LDLIBS) -o $@

random_light.o: random_light.f90
	$(FC) $(HDF5_FFLAGS) $(FFLAGS) -c $< -o $@

write_output.o: write_output.f90
	$(FC) $(HDF5_FFLAGS) $(FFLAGS) -c $< -o $@

fbm3d.o: fbm3d.f90 define.o random_light.o write_output.o
	$(FC) $(HDF5_FFLAGS) $(FFLAGS) -c $< -o $@

make_fbm3d.o: make_fbm3d.f90
	$(FC) $(HDF5_FFLAGS) $(FFLAGS) -c $< -o $@

define.o: define.f90
	$(FC) $(HDF5_FFLAGS) $(FFLAGS) -c $< -o $@

debug:
	$(MAKE) cleanall
	$(MAKE) USE_OPENMP=0 FFLAGS="-O0 -g -cpp -fcheck=all -Wall -Wextra"

clean:
	$(RM) *.o *.mod

cleanall: clean
	$(RM) make_fbm3d.x
