# fbm3d Fortran

The Fortran program generates a three-dimensional fractal density field using
the fractional Brownian motion algorithm described by Seon (2012) and
Seon & Draine (2016).

Author: Kwang-Il Seon

## Requirements

- A Fortran compiler such as GNU Fortran, Intel Fortran, or Intel `ifx`
- [CFITSIO](https://heasarc.gsfc.nasa.gov/fitsio/)
- [FFTW](https://www.fftw.org/), including the single-precision threaded library
- [HDF5](https://www.hdfgroup.org/) with Fortran support
- `pkg-config` is recommended for locating the libraries

## Build

GNU Fortran, OpenMP, and threaded FFTW are used by default.

```bash
make -j4
```

Select another compiler if needed:

```bash
make FC=ifort
```

Build without OpenMP:

```bash
make cleanall
make USE_OPENMP=0
```

Build with bounds checking and compiler warnings:

```bash
make debug
```

## Input

Copy and edit [`M005b040_001.in`](M005b040_001.in):

```fortran
&input
  iseed    = 0
  bvalue   = 0.4
  mach     = 0.5
  nx       = 256
  outfile  = 'M005b040_001.h5'
  out_mode = 1
/
```

Parameters:

- `iseed`: random seed; `0` selects an automatically generated seed
- `bvalue`: turbulence forcing parameter, mapped to `1/3`, `0.4`, or `1.0`
- `mach`: Mach number
- `nx`: cube dimension; it must be an even integer
- `outfile`: FITS or HDF5 output filename
- `out_mode`:
  - `1`: density field
  - `2`: density field and Gaussian-field power spectrum
  - `3`: mode 2 plus the final lognormal-field power spectrum

## Run

```bash
OMP_NUM_THREADS=8 ./make_fbm3d.x M005b040_001.in
```

OpenMP parallelizes field generation and reductions, while FFTW uses the same
thread count for transforms.

## Output formats

The filename extension selects the output format:

| Extension | Format |
|---|---|
| `.fits`, `.fits.gz` | FITS |
| `.h5`, `.hdf5` | HDF5 |

HDF5 files contain:

| Dataset | Availability |
|---|---|
| `data` | All output modes |
| `power_spectrum_gaussian` | Modes 2 and 3 |
| `power_spectrum_lognormal` | Mode 3 |

The HDF5 file attributes are `seed`, `bvalue`, `mach`, `slope_ln`,
`slope_gauss`, and `sigma_ln`. Datasets are stored as uncompressed `float32`
arrays to prioritize output speed.

For large cubes, prefer `.fits` or HDF5 during generation. Direct `.fits.gz`
output uses single-threaded gzip and can take substantially longer while often
providing little compression for random floating-point fields.

## Reproducibility

A fixed seed and fixed `OMP_NUM_THREADS` value produce the same output. Changing
the thread count changes the random realization while preserving the target
statistical properties.

## References

- Seon, K.-I. 2012, ApJL, 761, L17
- Seon, K.-I. & Draine, B. T. 2016, ApJ, 833, 201
