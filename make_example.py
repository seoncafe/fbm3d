#!/usr/bin/env python
# This python file demonstrates how to use fbm_lib.py
#
import numpy as np
from fbm_lib import *

#seed   = None
seed   = 9876540
nx     = 256
ny, nz = nx,nx

# Let's make sample density fields for mach numbers = 1, 2, 4, 6.
mach_arr = [1.0, 2.0, 4.0, 6.0]

for idx in np.arange(len(mach_arr)):
   mach = mach_arr[idx]
   fname = 'M%03d_%04d.fits.gz' % (mach*10, nx)
   print('making...',fname)
   a = fbm3d_ISM(nx,ny,nz,mach=mach)
   a.writeto(fname)
