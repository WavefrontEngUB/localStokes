#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pyfftw
from multiprocessing import cpu_count
pyfftw.config.NUM_THREADS = cpu_count()
nthr = cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"

def debye_int(Uin, f, Lx, Ly, lamb, z=0):
    """Focal light distribution in the scalar approximation uding the
    Debye integral. The input field is diffracted through the entrance
    pupil of the optical system assuming spherical coordinates. The
    Input:
        - Uin: Optical disturbance at the entrance pupil.
        - Pupil: Pupil function of the entrance pupil.
        - f: Focal length of the optical system.
        - Lx, Ly: half widths of the sampling region of Uin.
        - lamb: Wavelength of the light.
    Output:
        - Uf: Focal amplitude of the optical disturbance.
    """

    # Getting the number of points of the disturbance
    nx, ny = Uin.shape
    # Consistency check
    _nx, _ny = Pupil.shape
    if (_nx != nx) or (_ny != ny):
        raise ValueError("Shapes of Uin and Pupil must be the same")
    
    # Calculation of the cosine factor
    x = np.linspace(-Lx, Lx, nx)
    y = np.linspace(-Ly, Ly, ny)
    xx, yy = np.meshgrid(x, y)
    r2 = xx * xx + yy * yy
    sinth2 = r2 / f / f
    mask = sinth2 >= 1
    sinth2[mask] = 0
    costh = np.sqrt(1 - sinth2)

    # Setting up the transformation
    out = pyfftw.empty_aligned((nx, ny), dtype="complex128")
    inp = pyfftw.empty_aligned((nx, ny), dtype="complex128")
    product = Uin * (1 - mask) / costh
    if z != 0: product *= np.exp(2j * np.pi / lamb * z * costh)
    inp[:, :] = product.astype(np.complex128)
    fft_object = pyfftw.FFTW(inp, out, axes=(0, 1), threads=nthr)

    # Transformation
    Uf = fft_object()
    Uf = np.fft.fftshift(Uf)
    
    # Final coordinates
    rs = nx * lamb * f / 4 / Lx, ny * lamb * f / 4 / Ly

    return Uf, rs

def ricwolf_int(Ein, f, NA, Lx, Ly, lamb, z=0, n=1., P=None):
    """Wrapper function to perform a single calculation of the focal field
    using electromagnetic diffraction theory by Richards and Wolf."""
    # TODO: Better documentation
    Esp, kz = compute_ref_sphere_field(Ein, f, NA, Lx, Ly, lamb, n=n)
    nx, ny, _ = Esp.shape
    Lfx = nx*lamb*f/Lx
    Lfy = ny*lamb*f/Ly
    return compute_focal_field(Esp, z=z, kz=kz, P=P), (Lfx, Lfy)

def compute_ref_sphere_field(Ein, f, NA,  Lx, Ly, lamb, n=1.):
    """Computation of the field at the gaussian reference sphere, performing
    the transformation from polar to spherical coordinates. These function is
    handy if one needs to perform the calculation of the focal structure of the
    field at several different z planes. 
    Input:
        - Ein: Nx X Ny X 2 paraxial input field.
        - f: focal length of the optical system.
        - NA: Numerical aperture of the microscope
        - Lx, Ly: half widths of the input region.
        - lamb: wavelength of the light.
    Output:
        - E_sp: Nx X Ny X 3 electric field distribution at gaussian reference
            sphere.
        - kz: Wavevector in the z direction.
    """
    # TODO: Cleanup and better documentation
    # Getting the shape of the input field
    nx, ny, _ = Ein.shape
    # Calculation of the cosine factor
    y = np.linspace(-Ly, Ly, ny)
    x = np.linspace(-Lx, Lx, nx)
    xx, yy = np.meshgrid(y, x)
    sinthmax = NA/n
    thmax = np.arcsin(sinthmax)
    d = f*np.sin(thmax)
    d2 = d*d
    #d2 = n * n * f * f * NA * NA / ((n + NA) * (n - NA))
    r2 = xx * xx + yy * yy
    sinth2 = r2 / d2 
    mask = sinth2 >= sinthmax 
    sinth2[mask] = 0
    costh = np.sqrt(1 - sinth2)
    # Azimuthal angle
    phi = np.arctan2(yy, xx)
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)

    # Transformation to a three dimensional field
    E_sp = np.empty((nx, ny, 3), dtype=np.complex128)
    a1 = np.empty((nx, ny), dtype=np.complex128)
    a2 = np.empty((nx, ny), dtype=np.complex128)

    a1[:, :] = Ein[:, :, 0] * (-sinphi) + Ein[:, :, 1] * cosphi
    a2[:, :] = Ein[:, :, 0] * cosphi    + Ein[:, :, 1] * sinphi

    sqrtcos = np.sqrt(costh)
    E_sp[:, :, 0] = (1 - mask) * (a1 * (-sinphi) + a2 * costh * cosphi) /\
            sqrtcos
    E_sp[:, :, 1] = (1 - mask) * (a1 * cosphi + a2 * costh * sinphi) /\
            sqrtcos
    E_sp[:, :, 2] = (1 - mask) * (-a2 * np.sqrt(sinth2)) /\
            sqrtcos

    # Return also kz, the wavevector in the longitudinal direction
    kz = costh*2*np.pi/lamb*(1-mask)
    return E_sp, kz, d

def compute_focal_field(E_sp, z=0, kz=None, P=None):
    """Compute the focal field provided E_sp at a given z and with a given set
    of aberration coefficients in the Zernike's basis"""
    # TODO: Write better documentation and finish implementation of the 
    # FIXME: Change order of optional arguments
    # functions.
    """
    if P:
        # TODO: Implement pupil function
        pass
    """
    Ef = np.zeros_like(E_sp)
    nx, ny, _ = E_sp.shape

    # Planning the FFT, to be reused for each component of the field
    inp = pyfftw.empty_aligned((nx, ny), dtype="complex128")
    out = pyfftw.empty_aligned((nx, ny), dtype="complex128")
    fft_obj = pyfftw.FFTW(inp, out, axes=(0, 1), threads=nthr)
    for i in range(3):
        # Compute expontential only if necessary. Too slow otherwise (I think, 
        # maybe branch prediction error costs more...
        inp[:, :] = E_sp[:, :, i]*np.exp(1j*kz*z) if z != 0 else E_sp[:, :, i]
        Ef[:, :, i] = np.fft.fftshift(fft_obj())

    return Ef
