module output_mod
  use hdf5
  implicit none
  private
  public :: write_output
contains
!-------------------------------------------------------------------
  subroutine write_output(fname,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho, &
                          array,karray1,karray2)
  implicit none
  character(len=*), intent(in) :: fname
  integer, intent(in) :: iseed
  real, intent(in) :: array(:,:,:)
  real, intent(in), optional :: karray1(:,:,:),karray2(:,:,:)
  real, intent(in) :: bvalue,mach,slope_ln,slope_gauss,sigma_lnrho

  if (is_hdf5_filename(fname)) then
     call write_hdf5(fname,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho, &
                     array,karray1,karray2)
  else
     call write_fits(fname,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho, &
                     array,karray1,karray2)
  endif
  end subroutine write_output

!-------------------------------------------------------------------
  subroutine write_hdf5(fname,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho, &
                        array,karray1,karray2)
  implicit none
  character(len=*), intent(in) :: fname
  integer, intent(in) :: iseed
  real, intent(in) :: array(:,:,:)
  real, intent(in), optional :: karray1(:,:,:),karray2(:,:,:)
  real, intent(in) :: bvalue,mach,slope_ln,slope_gauss,sigma_lnrho

  integer(hid_t) :: file_id
  integer :: error

  call h5open_f(error)
  call check_hdf5(error,'initialize HDF5')
  call h5fcreate_f(trim(fname),H5F_ACC_TRUNC_F,file_id,error)
  call check_hdf5(error,'create HDF5 file')

  call write_hdf5_dataset(file_id,'data',array)
  if (present(karray1)) then
     call write_hdf5_dataset(file_id,'power_spectrum_gaussian',karray1)
  endif
  if (present(karray2)) then
     call write_hdf5_dataset(file_id,'power_spectrum_lognormal',karray2)
  endif

  call write_integer_attribute(file_id,'seed',iseed)
  call write_real_attribute(file_id,'bvalue',bvalue)
  call write_real_attribute(file_id,'mach',mach)
  call write_real_attribute(file_id,'slope_ln',slope_ln)
  call write_real_attribute(file_id,'slope_gauss',slope_gauss)
  call write_real_attribute(file_id,'sigma_ln',sigma_lnrho)

  call h5fclose_f(file_id,error)
  call check_hdf5(error,'close HDF5 file')
  call h5close_f(error)
  call check_hdf5(error,'close HDF5 library')
  end subroutine write_hdf5

!-------------------------------------------------------------------
  subroutine write_hdf5_dataset(file_id,name,array)
  implicit none
  integer(hid_t), intent(in) :: file_id
  character(len=*), intent(in) :: name
  real, intent(in) :: array(:,:,:)

  integer(hid_t) :: dataspace_id,dataset_id
  integer(hsize_t) :: dims(3)
  integer :: error

  dims = int(shape(array),kind=hsize_t)
  call h5screate_simple_f(3,dims,dataspace_id,error)
  call check_hdf5(error,'create dataspace for '//trim(name))
  call h5dcreate_f(file_id,trim(name),H5T_NATIVE_REAL,dataspace_id,dataset_id,error)
  call check_hdf5(error,'create dataset '//trim(name))
  call h5dwrite_f(dataset_id,H5T_NATIVE_REAL,array,dims,error)
  call check_hdf5(error,'write dataset '//trim(name))
  call h5dclose_f(dataset_id,error)
  call check_hdf5(error,'close dataset '//trim(name))
  call h5sclose_f(dataspace_id,error)
  call check_hdf5(error,'close dataspace for '//trim(name))
  end subroutine write_hdf5_dataset

!-------------------------------------------------------------------
  subroutine write_integer_attribute(object_id,name,value)
  implicit none
  integer(hid_t), intent(in) :: object_id
  character(len=*), intent(in) :: name
  integer, intent(in) :: value

  integer(hid_t) :: dataspace_id,attribute_id
  integer(hsize_t) :: dims(1)
  integer :: error

  dims = 1_hsize_t
  call h5screate_f(H5S_SCALAR_F,dataspace_id,error)
  call check_hdf5(error,'create attribute dataspace')
  call h5acreate_f(object_id,trim(name),H5T_NATIVE_INTEGER,dataspace_id,attribute_id,error)
  call check_hdf5(error,'create attribute '//trim(name))
  call h5awrite_f(attribute_id,H5T_NATIVE_INTEGER,value,dims,error)
  call check_hdf5(error,'write attribute '//trim(name))
  call h5aclose_f(attribute_id,error)
  call check_hdf5(error,'close attribute '//trim(name))
  call h5sclose_f(dataspace_id,error)
  call check_hdf5(error,'close attribute dataspace')
  end subroutine write_integer_attribute

!-------------------------------------------------------------------
  subroutine write_real_attribute(object_id,name,value)
  implicit none
  integer(hid_t), intent(in) :: object_id
  character(len=*), intent(in) :: name
  real, intent(in) :: value

  integer(hid_t) :: dataspace_id,attribute_id
  integer(hsize_t) :: dims(1)
  integer :: error

  dims = 1_hsize_t
  call h5screate_f(H5S_SCALAR_F,dataspace_id,error)
  call check_hdf5(error,'create attribute dataspace')
  call h5acreate_f(object_id,trim(name),H5T_NATIVE_REAL,dataspace_id,attribute_id,error)
  call check_hdf5(error,'create attribute '//trim(name))
  call h5awrite_f(attribute_id,H5T_NATIVE_REAL,value,dims,error)
  call check_hdf5(error,'write attribute '//trim(name))
  call h5aclose_f(attribute_id,error)
  call check_hdf5(error,'close attribute '//trim(name))
  call h5sclose_f(dataspace_id,error)
  call check_hdf5(error,'close attribute dataspace')
  end subroutine write_real_attribute

!-------------------------------------------------------------------
  subroutine check_hdf5(error,operation)
  implicit none
  integer, intent(in) :: error
  character(len=*), intent(in) :: operation

  if (error < 0) then
     write(*,*) 'HDF5 error: ',trim(operation)
     error stop 1
  endif
  end subroutine check_hdf5

!-------------------------------------------------------------------
  logical function is_hdf5_filename(fname)
  implicit none
  character(len=*), intent(in) :: fname
  character(len=:), allocatable :: normalized
  integer :: length

  normalized = lowercase(trim(fname))
  length = len(normalized)
  is_hdf5_filename = (length >= 3 .and. normalized(length-2:length) == '.h5')
  if (length >= 5) then
     is_hdf5_filename = is_hdf5_filename .or. normalized(length-4:length) == '.hdf5'
  endif
  end function is_hdf5_filename

!-------------------------------------------------------------------
  pure function lowercase(text) result(lower)
  implicit none
  character(len=*), intent(in) :: text
  character(len=len(text)) :: lower
  integer :: i,code

  lower = text
  do i=1,len(text)
     code = iachar(text(i:i))
     if (code >= iachar('A') .and. code <= iachar('Z')) then
        lower(i:i) = achar(code+iachar('a')-iachar('A'))
     endif
  enddo
  end function lowercase

!-------------------------------------------------------------------
  subroutine write_fits(fname,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho, &
                        array,karray1,karray2)
  implicit none
  character(len=*), intent(in) :: fname
  integer, intent(in) :: iseed
  real, intent(in) :: array(:,:,:)
  real, intent(in), optional :: karray1(:,:,:),karray2(:,:,:)
  real, intent(in) :: bvalue,mach,slope_ln,slope_gauss,sigma_lnrho

  integer :: status,unit,blocksize,bitpix,naxis,naxes(3),group
  integer :: fpixel,nelements
  logical :: simple,extend
  character(len=80) :: error_message

  call unlink(trim(fname))

  status = 0
  unit   = 1
  blocksize = 1
  call ftinit(unit,trim(fname),blocksize,status)

  simple = .true.
  bitpix = -32
  naxis  = 3
  naxes = shape(array)
  extend = .true.
  call ftphpr(unit,simple,bitpix,naxis,naxes,0,1,extend,status)

  group  = 1
  fpixel = 1
  nelements = product(naxes)
  call ftppre(unit,group,fpixel,nelements,array,status)
  call write_fits_attributes(unit,status,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho)

  if (present(karray1)) then
     naxes = shape(karray1)
     nelements = product(naxes)
     call ftiimg(unit,bitpix,naxis,naxes,status)
     call ftppre(unit,group,fpixel,nelements,karray1,status)
     call write_fits_attributes(unit,status,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho)
  endif

  if (present(karray2)) then
     naxes = shape(karray2)
     nelements = product(naxes)
     call ftiimg(unit,bitpix,naxis,naxes,status)
     call ftppre(unit,group,fpixel,nelements,karray2,status)
     call write_fits_attributes(unit,status,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho)
  endif

  call ftclos(unit,status)
  if (status /= 0) then
     call ftgerr(status,error_message)
     write(*,*) 'CFITSIO error: ',trim(error_message)
     error stop 1
  endif
  end subroutine write_fits

!-------------------------------------------------------------------
  subroutine write_fits_attributes(unit,status,iseed,bvalue,mach,slope_ln,slope_gauss,sigma_lnrho)
  implicit none
  integer, intent(in) :: unit,iseed
  integer, intent(inout) :: status
  real, intent(in) :: bvalue,mach,slope_ln,slope_gauss,sigma_lnrho

  call ftpkyj(unit,'ISEED',iseed,'Random Number seed',status)
  call ftpkye(unit,'B-VALUE',bvalue,-8,'b-value',status)
  call ftpkye(unit,'MACH',mach,-8,'mach',status)
  call ftpkye(unit,'SLOPE_LN',slope_ln,-8,'Power Spectral Index of Lognormal Field',status)
  call ftpkye(unit,'SLOPE_GA',slope_gauss,-8,'Power Spectral Index of Gaussian Field',status)
  call ftpkye(unit,'SIGMA_LN',sigma_lnrho,-8,'Standard Deviation in Ln scale',status)
  end subroutine write_fits_attributes
end module output_mod
