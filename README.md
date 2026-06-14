# fbm3d

## Overview
**fbm3d** is a code that generates lognormal or Gaussian random fields using the fractional Brownian motion (fBm) algorithm, designed to mimic the density field in the interstellar medium (ISM).

## Theory & References
The detailed algorithms are described in the following papers:
- [Seon (2012)](https://ui.adsabs.harvard.edu/search/q=author%3A%22Seon%22+year%3A2012+fBm) - Fundamental theory of the fBm algorithm
- [Lewis & Austin (2002)][lewis-austin-2002] - Lognormal distribution generation method

## Features

### LogNormalRandomField
Generates a 3D density field with the following characteristics:
- **Distribution**: Log-normal distribution
- **Spectrum**: Power-law-type power spectral density
- **Algorithm options**: 
  - `method=2`: Implementation based on [Lewis & Austin (2002)][lewis-austin-2002] algorithm (current default)
    - The current implementation differs from theirs but retains the core idea
  - `method=1`: Algorithm described in [Seon (2012)](https://ui.adsabs.harvard.edu/abs/2012ApJ...761L..17S/abstract)

### Key Parameters
```python
LogNormalRandomField(
    nx, ny, nz,           # Grid resolution
    seed=None,            # Random seed (auto-generated if None)
    mean=0.0,             # Mean of log(density)
    sigma=1.0,            # Standard deviation of log(density)
    verbose=True          # Print convergence progress
)
```

**Important**: `mean` and `sigma` represent the statistics of **log(density)**, NOT the density itself.

## Usage Examples

Detailed usage examples are provided in the following notebooks:
- [example.ipynb](example.ipynb) - Basic usage with fbm3d_ISM class
- [example2.ipynb](example2.ipynb) - Comparison of different algorithms
- [example3.ipynb](example3.ipynb) - LogNormalRandomField with 3D visualization and analysis

### Quick Start
```python
import numpy as np
from fbm_lib import LogNormalRandomField

# Generate a 256³ resolution lognormal field
field = LogNormalRandomField(
    nx=256, ny=256, nz=256,
    mean=0.0, sigma=1.0,
    verbose=True
)

# Check the generated random seed
print(f"Seed: {field.seed}")

# Save to FITS file
field.writeto('lognormal_256.fits.gz')
```

### Power Spectrum Analysis
```python
from fbm_lib import calculate_PSD, calculate_PSD_norm

kr, PSD = calculate_PSD(field.data)
ynorm = calculate_PSD_norm(field.data.shape, field.slope)

# Fit power-law: P(k) ∝ k^(-slope)
y = kr**(-field.slope) * ynorm
```

## Output

- **3D density field**: `field.data` (numpy array)
- **Log-transformed field**: `np.log(field.data)`
- **Power spectrum exponent**: `field.slope`
- **Random seed used**: `field.seed`

## Important Notes

⚠️ **This code will be updated without notice.**

For the ISM-specific class (`fbm3d_ISM`), the relationship between Mach number and log-density statistics follows:
```python
sigma_g = np.sqrt(np.log(1.0 + (bvalue * mach)**2))
```
where `bvalue` is the magnetic field parameter (typically 0.4) and `mach` is the Mach number.

[lewis-austin-2002]: https://ams.confex.com/ams/11AR11CP/webprogram/Paper42772.html
