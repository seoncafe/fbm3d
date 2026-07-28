# fbm3d

`fbm3d` generates two- and three-dimensional Gaussian or lognormal random
fields with power-law power spectra. It is designed primarily for constructing
fractal density fields that mimic the interstellar medium (ISM).

The repository provides:

- A Python implementation with Gaussian and iterative lognormal field generators
- An optimized Fortran implementation with OpenMP and threaded FFTW
- FITS and HDF5 output from both implementations
- Example notebooks for generation, analysis, and visualization

## Theory and references

The algorithms are described in:

- [Seon (2012, ApJL, 761, L17)](https://ui.adsabs.harvard.edu/abs/2012ApJ...761L..17S/abstract)
- [Seon & Draine (2016, ApJ, 833, 201)](https://ui.adsabs.harvard.edu/abs/2016ApJ...833..201S/abstract)
- [Lewis & Austin (2002)][lewis-austin-2002]

## Python implementation

### Requirements

- NumPy
- Astropy for FITS output
- h5py for HDF5 output
- Matplotlib for the plotting examples

### Available generators

- `GaussianRandomField2D`, aliased as `fbm2d`
- `GaussianRandomField`, aliased as `fbm3d`
- `LogNormalRandomField`
- `fbm3d_ISM`, also aliased as `fbm3d_lognormal_ISM`

`LogNormalRandomField` uses an iterative method based on the core idea of
Lewis & Austin (2002). In this class, `mean` and `sigma` describe
`log(density)`, not density itself.

### Quick start

```python
from fbm_lib import LogNormalRandomField

field = LogNormalRandomField(
    nx=256,
    ny=256,
    nz=256,
    mean=0.0,
    sigma=1.0,
    seed=12345,
    verbose=True,
)

print(field.seed)
print(field.data.shape)
```

The output format is selected by the filename extension:

```python
# FITS output
field.writeto("lognormal_256.fits.gz")

# HDF5 output
field.writeto("lognormal_256.h5")
```

Python HDF5 files contain a gzip-compressed dataset named `data`. Generation
parameters such as `seed`, `mean`, `sigma`, `kmin`, `kmax`, and `slope` are
stored as file attributes. `overwrite=False` prevents replacement of an
existing file.

The same output interface is available on `fbm2d`, `fbm3d`,
`LogNormalRandomField`, and `fbm3d_ISM` objects.

### ISM model

For `fbm3d_ISM`, the Gaussian log-density dispersion is

```python
sigma_g = np.sqrt(np.log(1.0 + (bvalue * mach)**2))
```

Typical `bvalue` choices are:

- `1/3`: solenoidal forcing
- `0.4`: natural mixture
- `1.0`: compressive forcing

Example:

```python
from fbm_lib import fbm3d_ISM

field = fbm3d_ISM(
    nx=256,
    ny=256,
    nz=256,
    mach=2.0,
    bvalue=0.4,
    seed=12345,
)
field.writeto("ism_mach2.h5")
```

### Power-spectrum analysis

```python
from fbm_lib import fbm3d, calculate_PSD, calculate_PSD_norm

field = fbm3d(128, 128, 128, slope=11.0 / 3.0, seed=12345)
kr, psd = calculate_PSD(field.data)
normalization = calculate_PSD_norm(field.data.shape, field.slope)
model_psd = kr**(-field.slope) * normalization
```

### Notebooks

- [example1.ipynb](example1.ipynb): basic `fbm3d_ISM` usage
- [example2.ipynb](example2.ipynb): algorithm comparison
- [example3.ipynb](example3.ipynb): lognormal-field visualization and analysis

## Fortran implementation

The implementation in [`fbm3d_fortran`](fbm3d_fortran/) uses:

- OpenMP for field generation and reductions
- Threaded single-precision FFTW for transforms
- Reduced temporary-array allocation
- Precomputed radial spectral amplitudes
- Conditional allocation of optional power-spectrum outputs

Full build and usage instructions are in
[`fbm3d_fortran/README.md`](fbm3d_fortran/README.md).

### Build and run

The Fortran version requires GNU or Intel Fortran, CFITSIO, FFTW, and HDF5 with
Fortran support.

```bash
cd fbm3d_fortran
make -j4
OMP_NUM_THREADS=8 ./make_fbm3d.x M005b040_001.in
```

Set `outfile` in the input namelist to select the format:

```fortran
outfile = 'density.fits'
```

or:

```fortran
outfile = 'density.h5'
```

Fortran HDF5 output is uncompressed for speed and contains:

- `data` for every output mode
- `power_spectrum_gaussian` for modes 2 and 3
- `power_spectrum_lognormal` for mode 3
- File attributes describing the seed and physical/spectral parameters

For large cubes, prefer uncompressed FITS or HDF5 during generation.
Single-threaded `.fits.gz` output can dominate the total runtime.

With a fixed seed and fixed OpenMP thread count, the Fortran output is
reproducible. Changing the thread count changes the realization while
preserving its target statistical properties.

[lewis-austin-2002]: https://ams.confex.com/ams/11AR11CP/webprogram/Paper42772.html

---

Last updated: 2026-07-28 13:22 KST
