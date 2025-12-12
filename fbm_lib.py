#!/usr/bin/env python
#--
#-- Written by K.I. Seon (2025.07.22) (originally 2022.09.19)
#--
#-- fbm2d and fbm3d give 2D and 3D fractal data, respectively, with mean = 0 and standard deviation = 1.
#-- fbm3d_ISM produces a 3D cube data to mimic the ISM fractal density, described by a lognormal distribution.
#-- See Seon (2012, ApJL, 761, L17), Seon & Draine (2016, ApJ, 833, 201), and Lewis & Austin (2002).
#--
#-- Usage: The followings show examples how to use this module.
# from fbm_lib import fbm2d, fbm3d, fbm3d_ISM
# a = fbm2d(128,128)
# a.writeto('a.fits.gz')
# plt.imshow(a.data, origin='lower')
# b = fbm3d(128,128,128)
# b.writeto('b.fits.gz')
# plt.imshow(b.data[0,:,:])
# plt.imshow(np.sum(b.data, axis=0))
# c = fbm3d_ISM(128,128,128, mach=2.0)
# c.writeto('c.fits.gz')
# plt.imshow(c.data[:,10,:], origin='lower')
# plt.imshow(np.sum(c.data, axis=1), origin='lower')

import numpy as np
from numpy.fft import fftfreq, fftn, ifftn, ifft2
import copy

#-----------------------------------------
def shift_center(img, centering=None):
    shape = np.shape(img)
    ndim  = len(shape)
    if centering == 1 or centering == 'min':
       loc = np.array(np.where(img == img.min())).reshape(ndim)
    elif centering == 2 or centering == 'max':
       loc = np.array(np.where(img == img.max())).reshape(ndim)
    else:
       return img

    lcen = np.array(shape) // 2
    ldel = lcen - loc

    arr = np.roll(img, shift = ldel, axis = range(ndim))
    return arr

#-----------------------------------------
def add_colorbar(im, width=None, pad=None, **kwargs):
    l, b, w, h = im.axes.get_position().bounds       # get boundaries
    width = width or 0.05 * w                        # get width of the colorbar
    pad = pad or width                               # get pad between im and cbar
    fig = im.axes.figure                             # get figure of image
    cax = fig.add_axes([l + w + pad, b, width, h])   # define cbar Axes
    return fig.colorbar(im, cax=cax, **kwargs)       # draw cbar

#-----------------------------------------
def zero_log(x):
    val = np.where(x > 0.0, np.log(x), 0.0)
    return val

#-----------------------------------------
def zero_div(a,b):
    val = np.where(np.abs(b) > 0.0, a/b, 0.0)
    return val

#-----------------------------------------
def calculate_PSD(data, **kwargs):
    ndim = len(data.shape)
    if ndim == 3:
       nx, ny, nz = data.shape
       kx = np.fft.fftfreq(nx) * nx
       ky = np.fft.fftfreq(ny) * ny
       kz = np.fft.fftfreq(nz) * nz
       kr = np.sqrt(np.add.outer(np.add.outer(kx**2, ky**2), kz**2))
       # Note that PSD[0,0,0] = mean * (nx*ny*nz) and kr[0,0,0] = 0.0
    elif ndim == 2:
       nx, ny = data.shape
       kx = np.fft.fftfreq(nx) * nx
       ky = np.fft.fftfreq(ny) * ny
       kr = np.sqrt(np.add.outer(kx**2, ky**2))
    PSD     = np.abs(np.fft.fftn(data))**2
    PSD     = PSD.ravel()
    norm    = 1.0/np.sum(PSD[1:])
    PSD[1:] = PSD[1:]*norm

    kr, indices, counts = np.unique(kr, return_inverse=True, return_counts=True)
    PSD      = np.bincount(indices.ravel(), weights = PSD)/counts
    return kr, PSD

#-----------------------------------------
def calculate_PSD_norm(shape=(64,64,64),slope=11./3.):
    ndim = len(shape)
    if ndim == 3:
       nx, ny, nz = shape
       kx = np.fft.fftfreq(nx) * nx
       ky = np.fft.fftfreq(ny) * ny
       kz = np.fft.fftfreq(nz) * nz
       kr = np.sqrt(np.add.outer(np.add.outer(kx**2, ky**2), kz**2))
       # Note that PSD[0,0,0] = mean * (nx*ny*nz) and kr[0,0,0] = 0.0
    elif ndim == 2:
       nx, ny = data.shape
       kx = np.fft.fftfreq(nx) * nx
       ky = np.fft.fftfreq(ny) * ny
       kr = np.sqrt(np.add.outer(kx**2, ky**2))
    norm = 1.0/np.sum(kr.ravel()[1:]**(-slope))
    return norm

#-----------------------------------------
class GaussianRandomField2D:
  def __init__(self,nx=64,ny=64,slope=2.8,mean=0.0,sigma=1.0,kmin=None,kmax=None,output_Ak=False,
               seed=None,gaussian_amplitude=False,dtype='float32',centering=None):

     #--- a very simple random seed generation (2025.12.12)
     if seed == None: seed = int(np.random.rand() * (2**32 - 1))
     np.random.seed(seed=seed)

     phi = np.random.random((nx,ny)) * 2.0 * np.pi
     ang = np.zeros((nx,ny))
     ang[1:,1:] = phi[1:,1:] - phi[:0:-1,:0:-1]
     ang[0, 1:] = phi[0, 1:] - phi[0, :0:-1]
     ang[1:,0]  = phi[1:,0]  - phi[:0:-1,0]
     ang[0,0]   = 0.0
     del phi

     # Fourier Coefficient for Flat Random Field
     Ak_flat = np.cos(ang) + np.sin(ang)*1j

     kx = fftfreq(nx) * nx
     ky = fftfreq(ny) * ny
     kr = np.sqrt(np.add.outer(kx**2, ky**2))

     if kmin == None: kmin = 0.0
     if kmax == None: kmax = np.max(kr)

     # Apodize the Power Spectral Density (PSD).
     PSD      = np.where(np.greater(kr, 0.0), kr**(-np.abs(slope)), 0.0)
     PSD[0,0] = 0.0
     if kmin > 0.0 or kmax < np.max(kr):
         PSD   = np.where((kr >= kmin) & (kr <= kmax), PSD, 0.0)
     const    = 1.0 / np.sum(PSD)
     PSD     *= const
     Ak       = Ak_flat * np.sqrt(PSD)
     Ak[0,0]  = 0.0
     del PSD

     #--- To be tested.
     if gaussian_amplitude == True:
        gauss = np.random.normal(0.0,1.0,size=(nx,ny))
        Ak = Ak * gauss

     # norm = forward for inverse transform gives unscaled result, gives the standard deviation of 1.
     #      = backward (default) for inverse transform gives a result scaled by 1/N.
     # imaginary part should be zero!
     img = np.real(ifft2(Ak, norm='forward')) * sigma + mean

     #--- centering
     if centering != None: img = shift_center(img, centering=centering)

     #--- float32
     if dtype != 'float64': img  = np.float32(img)
     self.data   = img
     self.kmin   = kmin
     self.kmax   = kmax
     self.mean   = mean
     self.sigma  = sigma
     self.slope  = slope
     self.seed   = seed
     if output_Ak == True: self.Ak = Ak

  def centering(self,centering):
     self.data = shift_center(self.data, centering)

  def copy(self):
      return copy.deepcopy(self)

  def writeto(self,fits_file=None,overwrite=True):
     from astropy.io import fits
     if fits_file != None:
        fits_file = fits_file.replace('.fits.gz','').replace('.fits','')+'.fits.gz'
        hdr          = fits.Header()
        hdr['seed']  = (self.seed,  'seed')
        hdr['mean']  = (self.mean,  'mean')
        hdr['sigma'] = (self.sigma, 'standard deviation')
        hdr['kmin']  = (self.kmin,  'wavenumber min')
        hdr['kmax']  = (self.kmin,  'wavenumber max')
        hdr['slope'] = (self.slope, 'power spectrum slope')
        hdu = fits.PrimaryHDU(self.data, header=hdr)
        hdu.writeto(fits_file,overwrite=overwrite)

#-----------------------------------------
class GaussianRandomField:
  def __init__(self,nx=64,ny=64,nz=64,slope=11./3.,mean=0.0,sigma=1.0,kmin=None,kmax=None,output_Ak=False,
               seed=None,gaussian_amplitude=False,dtype='float32',centering=None):
  
     #--- a very simple random seed generation (2025.12.12)
     if seed == None: seed = int(np.random.rand() * (2**32 - 1))
     np.random.seed(seed=seed)
  
     phi = np.random.random((nx,ny,nz)) * 2.0 * np.pi
     ang = np.zeros((nx,ny,nz))
     ang[1:,1:,1:] = phi[1:,1:,1:] - phi[:0:-1,:0:-1,:0:-1]
     ang[1:,1:,0]  = phi[1:,1:,0]  - phi[:0:-1,:0:-1,0]
     ang[1:,0,1:]  = phi[1:,0,1:]  - phi[:0:-1,0,:0:-1]
     ang[0,1:,1:]  = phi[0,1:,1:]  - phi[0,:0:-1,:0:-1]
     ang[1:,0,0]   = phi[1:,0,0]   - phi[:0:-1,0,0]
     ang[0,1:,0]   = phi[0,1:,0]   - phi[0,:0:-1,0]
     ang[0,0,1:]   = phi[0,0,1:]   - phi[0,0,:0:-1]
     ang[0,0,0]    = 0.0
     del phi

     # Fourier Coefficient for Flat Random Field
     Ak_flat = np.cos(ang) + np.sin(ang)*1j
  
     kx = fftfreq(nx) * nx
     ky = fftfreq(ny) * ny
     kz = fftfreq(nz) * nz
     kr = np.sqrt(np.add.outer(np.add.outer(kx**2, ky**2), kz**2))

     if kmin == None: kmin = 0.0
     if kmax == None: kmax = np.max(kr)

     # Apodize the Power Spectral Density (PSD).
     PSD        = np.where(np.greater(kr, 0.0), kr**(-np.abs(slope)), 0.0)
     PSD[0,0,0] = 0.0
     if kmin > 0.0 or kmax < np.max(kr):
         PSD   = np.where((kr >= kmin) & (kr <= kmax), PSD, 0.0)
     const      = 1.0 / np.sum(PSD)
     PSD       *= const
     Ak         = Ak_flat * np.sqrt(PSD)
     Ak[0,0,0]  = 0.0
     del PSD

     #--- To be tested.
     if gaussian_amplitude == True:
        gauss = np.random.normal(0.0,1.0,size=(nx,ny,nz))
        Ak = Ak * gauss

     # norm = forward for inverse transform gives unscaled result, gives the standard deviation of 1.
     #      = backward (default) for inverse transform gives a result scaled by 1/N.
     # imaginary part should be zero!
     img    = np.real(ifftn(Ak, norm='forward')) * sigma + mean
  
     #--- centering
     if centering != None: img = shift_center(img, centering=centering)

     #--- float32
     if dtype != 'float64': img  = np.float32(img)
     self.data   = img
     self.kmin   = kmin
     self.kmax   = kmax
     self.mean   = mean
     self.sigma  = sigma
     self.slope  = slope
     self.seed   = seed
     if output_Ak == True: self.Ak = Ak

  def centering(self,centering):
     self.data = shift_center(self.data, centering)

  def copy(self):
      return copy.deepcopy(self)

  def writeto(self,fits_file=None,overwrite=True):
     from astropy.io import fits
     if fits_file != None:
        fits_file = fits_file.replace('.fits.gz','').replace('.fits','')+'.fits.gz'
        hdr          = fits.Header()
        hdr['seed']  = (self.seed,  'seed')
        hdr['mean']  = (self.mean,  'mean')
        hdr['sigma'] = (self.sigma, 'standard deviation')
        hdr['kmin']  = (self.kmin,  'wavenumber min')
        hdr['kmax']  = (self.kmin,  'wavenumber max')
        hdr['slope'] = (self.slope, 'power spectrum slope')
        hdu = fits.PrimaryHDU(self.data, header=hdr)
        hdu.writeto(fits_file,overwrite=overwrite)

#-----------------------------------------
class LogNormalRandomField:
    # 2025.12.12: The gaussian_amplitude option needs testing and should not be used for now.
    # 2025.07.18
    def __init__(self,nx=64,ny=64,nz=64,slope=11./3., mean=0.0, sigma=1.0, kmin=None, kmax=None, verbose=False,
                 seed=None,gaussian_amplitude=None,dtype='float32',centering=None):

        #--- a very simple random seed generation (2025.12.12)
        if seed == None: seed = int(np.random.rand() * (2**32 - 1))
        np.random.seed(seed=seed)

        iter_max      = 30
        converge_tol  = 0.005
        grad_fraction = 0.5
        # dop = Degree of polynomial (dop should be greater than or equal to 3.)
        dop           = 5

        # standard deviation of log-normal distribution.
        var_ln   = (np.exp(sigma**2)-1.0)*np.exp(2.0*mean + sigma**2)
        sigma_ln = np.sqrt(var_ln)
        mean     = 0.0
        mean_ln  = np.exp(mean + sigma**2/2.0)

        #--- Random Phases for Real Data ---
        phi = np.random.random((nx,ny,nz)) * 2.0 * np.pi
        ang = np.zeros((nx,ny,nz))
        ang[1:,1:,1:] = phi[1:,1:,1:] - phi[:0:-1,:0:-1,:0:-1]
        ang[1:,1:,0]  = phi[1:,1:,0]  - phi[:0:-1,:0:-1,0]
        ang[1:,0,1:]  = phi[1:,0,1:]  - phi[:0:-1,0,:0:-1]
        ang[0,1:,1:]  = phi[0,1:,1:]  - phi[0,:0:-1,:0:-1]
        ang[1:,0,0]   = phi[1:,0,0]   - phi[:0:-1,0,0]
        ang[0,1:,0]   = phi[0,1:,0]   - phi[0,:0:-1,0]
        ang[0,0,1:]   = phi[0,0,1:]   - phi[0,0,:0:-1]
        ang[0,0,0]    = 0.0
        del phi

        # Fourier Coefficient for Flat Random Field
        Ak_flat = np.cos(ang) + np.sin(ang)*1j

        #--- Basic Setup
        kx = fftfreq(nx) * nx
        ky = fftfreq(ny) * ny
        kz = fftfreq(nz) * nz
        kr = np.sqrt(np.add.outer(np.add.outer(kx**2, ky**2), kz**2))

        kr_x, indices, counts = np.unique(kr, return_inverse=True, return_counts=True)
        indices = indices.ravel()

        if kmin == None: kmin = 0.0
        if kmax == None: kmax = np.max(kr_x)

        # Apodize the Power Spectral Density (PSD).
        PSD0     = np.where(np.greater(kr, 0.0), kr**(-np.abs(slope)), 0.0)
        const    = sigma**2 / np.sum(PSD0.ravel()[1:])
        PSD0    *= const
        PSD0[0,0,0] = 0.0
        Ak        = Ak_flat * np.sqrt(PSD0)
        Ak[0,0,0] = 0.0
        # To ensure that the field has a mean value of 'mean'
        #PSD0[0,0,0] = mean**2
        #Ak[0,0,0]   = mean

        #--- To be tested. (2025.12.12)
        if gaussian_amplitude == True:
           gauss = np.random.normal(0.0,1.0,size=(nx,ny,nz))
           Ak = Ak * gauss

        # Power Spectral Density of np.exp(data)
        PSD0_ln        = PSD0 * (sigma_ln/sigma)**2
        PSD0_ln[0,0,0] = mean_ln**2

        log_kr     = zero_log(kr)
        log_kr_x   = zero_log(kr_x)
        ref_logPSD = zero_log(PSD0_ln)

        # norm = forward for inverse transform gives unscaled result, gives the standard deviation of 1.
        #      = backward (default) for inverse transform gives a result scaled by 1/N.
        # imaginary part should be zero!
        data = np.real(ifftn(Ak, norm='forward'))
        #print('data.mean, data.std ==', data.mean(), data.std())

        # Loop begins here
        convergence, iter = 1.0, 0
        while convergence > converge_tol and iter < iter_max:
            exp_data  = np.exp(data)
            PSD       = np.abs(fftn(exp_data))**2
            cc         = sigma_ln**2 / np.sum(PSD.ravel()[1:])
            PSD        *= cc
            PSD[0,0,0] = mean_ln**2
            PSD_x     = np.bincount(indices, weights = PSD.ravel())/counts
            log_PSD_x = zero_log(PSD_x)
            ####################
            #if iter == 0:
            #    self.PSD_x = PSD_x[np.newaxis,:]
            #else:
            #    self.PSD_x = np.append(self.PSD_x, PSD_x[np.newaxis,:], axis=0)
            ####################

            # Polynomial Fitting
            p_coeff    = np.polyfit(log_kr_x[1:], log_PSD_x[1:], dop)
            fit_logPSD = np.polyval(p_coeff, log_kr)

            # Correction of Log(Density) Field.
            # (1) This does not seem to work well. Very slow convergence!
            #corr       = grad_fraction * (PSD0_ln - np.exp(fit_logPSD))
            #corr       = np.where(np.greater(kr, 0.0), corr, 0.0)
            #PSD        = np.where(np.greater(PSD0 + corr, 0.0), PSD0 + corr, PSD0)
            # (2) in logarithmic scale
            ## The following method gives almost perfect PSD. However, Lognormal PDF is not guaranteed.
            ##corr      = np.where(np.greater(kr, 0.0), eta * (ref_logPSD - zero_log(PSD)), 0.0)
            # (3) in logarithmic scale. This is the best method! (2025.07.20)
            corr      = np.where(np.greater(kr, 0.0), grad_fraction * (ref_logPSD - fit_logPSD), 0.0)
            PSD        = PSD0 * np.exp(corr)

            PSD[0,0,0] = 0.0
            cc         = sigma**2 / np.sum(PSD)
            PSD        *= cc

            Ak        = Ak_flat * np.sqrt(PSD)
            Ak[0,0,0] = 0.0
            data      = np.real(ifftn(Ak, norm='forward'))

            # Estimate convergence
            convergence = np.average(zero_div(abs(PSD0 - PSD), PSD))
            if convergence > converge_tol: PSD0  = PSD.copy()
            iter += 1

            # Verbose Messages
            if verbose:
                converge_level = (1.0 - convergence) * 100.0
                print('iteration = %3d / convergence = %.2f %%' % (iter, converge_level))

        #---
        if kmin > 0.0 or kmax < np.max(kr_x):
            PSD    = np.where((kr >= kmin) & (kr <= kmax), PSD, 0.0)
            cc     = sigma**2 / np.sum(PSD)
            PSD   *= cc
            Ak     = Ak_flat * np.sqrt(PSD)
            data   = np.real(ifftn(Ak, norm='forward'))

        #--- float32
        if dtype != 'float64': data  = np.float32(data)

        #--- centering
        if centering != None: data = shift_center(data, centering=centering)

        # Return the Final data
        self.seed  = seed
        self.mean  = mean
        self.sigma = sigma
        self.slope = slope
        self.kmin  = kmin
        self.kmax  = kmax
        self.data  = np.exp(data + mean)

    def centering(self,centering):
       self.data = shift_center(self.data, centering)

    def copy(self):
        return copy.deepcopy(self)

    def writeto(self,fits_file=None,overwrite=True):
        from astropy.io import fits
        if fits_file != None:
           fits_file = fits_file.replace('.fits.gz','').replace('.fits','')+'.fits.gz'
        hdr          = fits.Header()
        hdr['seed']  = (self.mean,  'seed')
        hdr['mean']  = (self.mean,  'mean')
        hdr['sigma'] = (self.sigma, 'standard deviation')
        hdr['kmin']  = (self.kmin,  'wavenumber min')
        hdr['kmax']  = (self.kmin,  'wavenumber max')
        hdr['slope'] = (self.slope, 'power spectrum slope')
        hdu = fits.PrimaryHDU(self.data, header=hdr)
        hdu.writeto(fits_file,overwrite=overwrite)

#-----------------------------------------
class fbm3d_ISM:
  def __init__(self,nx=64,ny=64,nz=64,mach=1.0,bvalue=0.4,method=2,normalize=False,kmin=None,kmax=None,verbose=False,
               seed=None,gaussian_amplitude=False,dtype='float32',centering=None):

     ##--- generate random realization
     #if seed == None: seed = int(np.random.rand() * (2**32 - 1))
     #np.random.seed(seed=seed)

     par = np.array([ 2.841e-01, -9.168e-01, -9.334e-01,  1.221e+00, -2.546e-01,
                      8.173e-01,  5.994e-01,  1.326e+00, -1.125e+00,  2.119e-01,
                      5.019e-02, -1.468e-01, -3.838e-01,  2.970e-01, -5.417e-02,
                     -4.428e-03,  1.186e-02,  3.111e-02, -2.375e-02,  4.329e-03]).reshape(4,5)

     # See Seon (2012, ApJL, 761, L17) and Seon & Draine (2016, ApJ, 833, 201)
     # bvalue = 1/3, 0.4, 1.0 for solenoidal, natural mixing, and compressive modes
     if bvalue < 0.4:
        bvalue = 1.0/3.0
     elif bvalue > 0.4:
        bvalue = 1.0

     # _g   refers to GaussianRandomField
     # _ln  refers to LogNormalRandomField
     sigma_g  = np.sqrt(np.log(1.0+(bvalue*mach)**2))
     mean_g   = 0.0
     mean_ln  = np.exp(mean_g + sigma_g**2/2.0)
     sigma_ln = np.sqrt((np.exp(sigma_g**2)-1.0)*np.exp(2.0*mean_g + sigma_g**2))

     if bvalue <= 0.4:
        slope_ln = 3.81*mach**(-0.16)
     else:
        slope_ln = 3.81*mach**(-0.16) + 0.6

     bcoeff = np.zeros(4)
     for j in np.arange(4):
        bcoeff[j] = np.sum(par[j,:] * sigma_g**np.arange(5))
     slope_g = np.sum(bcoeff[:] * slope_ln**np.arange(4))

     if method == 1:
        img = GaussianRandomField(nx=nx,ny=ny,nz=nz,kmin=kmin,kmax=kmax,slope=slope_g,seed=seed,
                                  gaussian_amplitude=gaussian_amplitude,dtype=dtype,centering=centering)
        self.data = np.exp(img.data * sigma_g)
     else:
        img = LogNormalRandomField(nx=nx,ny=ny,nz=nz,kmin=kmin,kmax=kmax,sigma=sigma_g,slope=slope_ln,seed=seed,verbose=verbose,
                                  gaussian_amplitude=gaussian_amplitude,dtype=dtype,centering=centering)
        self.data = img.data

     self.seed        = img.seed
     self.mach        = mach
     self.bvalue      = bvalue
     self.mean_g      = mean_g
     self.sigma_g     = sigma_g
     self.slope_g     = slope_g
     self.mean_ln     = mean_ln
     self.sigma_ln    = sigma_ln
     self.slope_ln    = slope_ln
     self.kmin        = img.kmin
     self.kmax        = img.kmax
     if dtype != 'float64':
        self.data = np.float32(self.data)
     if normalize == True:
        self.data = self.data/np.mean(self.data)

  def centering(self,centering):
     self.data = shift_center(self.data, centering)

  def copy(self):
      return copy.deepcopy(self)

  def writeto(self,fits_file=None,overwrite=True):
     from astropy.io import fits
     if fits_file != None:
        fits_file = fits_file.replace('.fits.gz','').replace('.fits','')+'.fits.gz'
        hdr             = fits.Header()
        hdr['seed']     = (self.seed,        'Random Number Seed')
        hdr['mach']     = (self.mach,        'Mach number')
        hdr['bvalue']   = (self.bvalue,      'b (1/3=solenoidal,0.4=natural,1.0=compressive)')
        hdr['mean_g']   = (self.mean_g,      'mean of Gaussian Random field')
        hdr['sigma_g']  = (self.sigma_g,     'stddev of Gaussian Random field')
        hdr['slope_g']  = (self.slope_g,     'PSD slope of Gaussian Random field')
        hdr['mean_ln']  = (self.mean_ln,     'mean of LogNormal Random field')
        hdr['sigma_ln'] = (self.sigma_ln,    'stddev of LogNormal Random field')
        hdr['slope_ln'] = (self.slope_ln,    'PSD slope of LogNormal Random field')
        hdr['kmin']     = (self.kmin,        'wavenumber min')
        hdr['kmax']     = (self.kmin,        'wavenumber max')
        hdu = fits.PrimaryHDU(self.data, header=hdr)
        hdu.writeto(fits_file,overwrite=overwrite)
        print('a FITS file saved: ', fits_file)

fbm3d_lognormal_ISM = fbm3d_ISM
fbm3d               = GaussianRandomField
fbm2d               = GaussianRandomField2D
