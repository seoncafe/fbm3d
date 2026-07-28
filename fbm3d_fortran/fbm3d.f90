   subroutine fbm3d(iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho,nx,ny,nz,outfile,out_mode)
!-------------------------------------------------
! Generate Fractal Density in 3D using fractional Brownian motion algorithm.
! 2009/04, Kwangil Seon
!
! 2013-09-17, slightly modified.
!
! 2026-07-28, performance and memory optimization:
!   - Parallelized field generation, reductions, and transforms with OpenMP
!     and threaded single-precision FFTW.
!   - Changed loop ordering for contiguous Fortran memory access.
!   - Replaced radial-power calculations at every grid cell with a
!     precomputed amplitude table.
!   - Removed the ang and gauss_k temporary cubes.
!   - Allocated Pk1 and Pk2 only when requested by out_mode.
!   - Combined normalization and the lognormal transform into one array pass.
!   - Added deterministic random-number streams for each OpenMP thread when
!     the seed and thread count are fixed.
!-------------------------------------------------

   use define, only : twopi, wp, i8b
   use random
   use output_mod, only : write_output
#ifdef _OPENMP
   use iso_c_binding, only : c_int
   use omp_lib, only : omp_get_max_threads
#endif
   implicit none

#ifdef _OPENMP
   interface
      function fftwf_init_threads() bind(C,name='fftwf_init_threads') result(status)
         import :: c_int
         integer(c_int) :: status
      end function fftwf_init_threads
      subroutine fftwf_plan_with_nthreads(nthreads) bind(C,name='fftwf_plan_with_nthreads')
         import :: c_int
         integer(c_int), value :: nthreads
      end subroutine fftwf_plan_with_nthreads
      subroutine fftwf_cleanup_threads() bind(C,name='fftwf_cleanup_threads')
      end subroutine fftwf_cleanup_threads
   end interface
#endif

   integer, intent(in) :: nx,ny,nz,out_mode
   integer, intent(inout) :: iseed
   real, intent(in)  :: bvalue,mach,slope_ln,slope_gauss,sigma_lnrho
   character(len=100), intent(in) :: outfile

   integer, parameter :: FFTW_ESTIMATE = 64
   integer(kind=i8b) :: plan

   complex(kind=wp), allocatable, dimension(:,:,:) :: Ak
   real(kind=wp), allocatable, dimension(:,:,:) :: arr,phi,Pk1,Pk2
   real(kind=wp), allocatable, dimension(:) :: radial_amplitude

   real(kind=wp) :: Anorm,kscale,stdev,mean,weight
   integer :: kx2(nx/2+1),ky2(ny),kz2(nz)
   integer :: i,j,k,ii,jj,kk,r2,max_r2,phase_seed,amplitude_seed
#ifdef _OPENMP
   integer :: threads_ok
#endif

   phase_seed = iseed
   if (iseed == 0) then
      amplitude_seed = 0
   else
      amplitude_seed = ieor(iseed,104729)
   endif

#ifdef _OPENMP
   threads_ok = fftwf_init_threads()
   if (threads_ok == 0) error stop 'FFTW thread initialization failed.'
   call fftwf_plan_with_nthreads(omp_get_max_threads())
#endif

   kscale = 1.0_wp/real(nx,wp)/real(ny,wp)/real(nz,wp)

   do i=1,nx/2+1
     kx2(i) = (i-1)**2
   enddo
   do j=1,ny/2+1
     ky2(j) = (j-1)**2
   enddo
   do k=1,nz/2+1
     kz2(k) = (k-1)**2
   enddo
   do j=1,ny/2-1
     ky2(ny/2+1+j) = (-ny/2+j)**2
   enddo
   do k=1,nz/2-1
     kz2(nz/2+1+k) = (-nz/2+k)**2
   enddo

! Precompute the power-law amplitude for every possible squared radius.
   max_r2 = maxval(kx2)+maxval(ky2)+maxval(kz2)
   allocate(radial_amplitude(0:max_r2))
   radial_amplitude(0) = 0.0_wp
!$omp parallel do schedule(static)
   do r2=1,max_r2
     radial_amplitude(r2) = real(r2,wp)**(-slope_gauss/4.0_wp)
   enddo
!$omp end parallel do

   allocate(phi(nx,ny,nz))
!$omp parallel private(i,j,k)
   call init_random_seed(phase_seed)
!$omp do collapse(2) schedule(static)
   do k=1,nz
   do j=1,ny
   do i=1,nx
     phi(i,j,k) = real(rand_number(),wp)
   enddo
   enddo
   enddo
!$omp end do
!$omp end parallel

   allocate(Ak(nx/2+1,ny,nz))
!$omp parallel do collapse(2) schedule(static) private(i,ii,jj,kk,r2)
   do k=1,nz
   do j=1,ny
     kk = modulo(nz-(k-1),nz)+1
     jj = modulo(ny-(j-1),ny)+1
   do i=1,nx/2+1
     ii = modulo(nx-(i-1),nx)+1
     r2 = kx2(i)+ky2(j)+kz2(k)
     Ak(i,j,k) = cmplx( &
         cos(twopi*(phi(i,j,k)-phi(ii,jj,kk))), &
         sin(twopi*(phi(i,j,k)-phi(ii,jj,kk))), kind=wp) &
         * radial_amplitude(r2)
   enddo
   enddo
   enddo
!$omp end parallel do
   deallocate(phi,radial_amplitude)

! Ak(1,1,1) -> average value of the resulting field
   Ak(1,1,1) = cmplx(0.0_wp,0.0_wp,kind=wp)

!----------------------------------------------------------------
   Anorm = 0.0_wp
!$omp parallel do collapse(2) schedule(static) private(i,weight) reduction(+:Anorm)
   do k=1,nz
   do j=1,ny
   do i=1,nx/2+1
      weight = 2.0_wp
      if (i == 1 .or. i == nx/2+1) weight = 1.0_wp
      Anorm = Anorm+weight*(real(Ak(i,j,k),wp)**2+aimag(Ak(i,j,k))**2)
   enddo
   enddo
   enddo
!$omp end parallel do
   write(*,*) 'K_norm',Anorm
!$omp parallel workshare
   Ak = Ak/sqrt(Anorm)
!$omp end parallel workshare

!----------------------------------------------------------------
!$omp parallel private(i,j,k)
   call init_random_seed(amplitude_seed)
!$omp do collapse(2) schedule(static)
   do k=1,nz
   do j=1,ny
   do i=1,nx/2+1
      Ak(i,j,k) = Ak(i,j,k)*real(rand_gauss(),wp)
   enddo
   enddo
   enddo
!$omp end do
!$omp end parallel

!----------------------------------------------------------------
   if (out_mode >= 2) then
      allocate(Pk1(nx/2+1,ny,nz))
!$omp parallel workshare
      Pk1 = abs(Ak)**2
!$omp end parallel workshare
   endif

!----------------------------------------------------------------
   allocate(arr(nx,ny,nz))
   call sfftw_plan_dft_c2r_3d(plan,nx,ny,nz,Ak,arr,FFTW_ESTIMATE)
   call sfftw_execute_dft_c2r(plan,Ak,arr)
   call sfftw_destroy_plan(plan)

   mean = 0.0_wp
!$omp parallel do collapse(2) schedule(static) private(i) reduction(+:mean)
   do k=1,nz
   do j=1,ny
   do i=1,nx
      mean = mean+arr(i,j,k)
   enddo
   enddo
   enddo
!$omp end parallel do
   mean = mean/real(size(arr),wp)

   stdev = 0.0_wp
!$omp parallel do collapse(2) schedule(static) private(i) reduction(+:stdev)
   do k=1,nz
   do j=1,ny
   do i=1,nx
      stdev = stdev+(arr(i,j,k)-mean)**2
   enddo
   enddo
   enddo
!$omp end parallel do
   stdev = sqrt(stdev/real(size(arr),wp))
   write(*,*) 'MEAN, STDEV',mean,stdev

! Transform to lognormal distribution
!$omp parallel workshare
   arr = exp((arr-mean)*(sigma_lnrho/stdev))
!$omp end parallel workshare

!----------------------------------------------------------------
   if (out_mode == 3) then
      allocate(Pk2(nx/2+1,ny,nz))
      call sfftw_plan_dft_r2c_3d(plan,nx,ny,nz,arr,Ak,FFTW_ESTIMATE)
      call sfftw_execute_dft_r2c(plan,arr,Ak)
      call sfftw_destroy_plan(plan)
!$omp parallel workshare
      Pk2 = abs(Ak*kscale)**2
!$omp end parallel workshare
   endif

!----------------------------------------------------------------
   select case (out_mode)
   case (1)
      call write_output(outfile,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho,arr)
   case (2)
      call write_output(outfile,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho,arr,Pk1)
   case (3)
      call write_output(outfile,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho,arr,Pk1,Pk2)
   end select
   deallocate(arr)
   if (allocated(Pk1)) deallocate(Pk1)
   if (allocated(Pk2)) deallocate(Pk2)
   deallocate(Ak)

#ifdef _OPENMP
   call fftwf_cleanup_threads()
#endif

   end subroutine fbm3d
