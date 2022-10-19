#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pyfftw
from multiprocessing import cpu_count
from propygator.errors import DimensionError
pyfftw.config.NUM_THREADS = cpu_count()
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

def freespace(Uin, Lx, Ly, z, lamb, prop_method="auto"):
    """Free space propagation function. Selects the correct propagation
    method (plane wave spectrum or Fresnel diffractoin) based on the
    dimensions of the sampling window (2Lx, 2Ly), the distance z and
    the wavelength lamb.
    Input:
        - Uin: Optical disturbance or electric field component
        - Lx, Ly: half widths of the sampling window
        - z: distance between the input and output planes
        - lamb: Wavelength of the light.
    Output:
        -Uout: Optical disturbance or electric field component at
            the output plane.
        -(Lox, Loy): Half widths of the output sampling window.
    """
    # Getting the dimensions of the sampling region
    try:
        (Nx, Ny) = Uin.shape
    except:
        DimensionError("Uin must be a 2D array")

    # Selecting the correct regime
    if prop_method == "auto":
        Lox = Nx * lamb * z / 4 / Lx  # If squared sampling, equal for y
        case = "fresnel" if  Lox >= Lx else "planewave"
    elif prop_method == "fresnel":
        case = "fresnel"
    elif prop_method == "planewave":
        case = "planewave"
    else:
        raise ValueError("Propagation method not recognized")

    # Preparing the transformation
    inp = pyfftw.empty_aligned((Nx, Ny), dtype="complex128")
    out = pyfftw.empty_aligned((Nx, Ny), dtype="complex128")
    fft_object = pyfftw.FFTW(inp, out, axes=(0, 1))

    # Propagating
    if case == "fresnel":
        ct = -1j/lamb/z
        x = np.linspace(-Lx, Lx, Nx)
        y = np.linspace(-Ly, Ly, Ny)
        xx, yy = np.meshgrid(y, x)
        # Kernel of the transformation
        kernel = np.exp(1j * np.pi / lamb / z * (xx * xx + yy * yy))
        inp[:, :] = fftshift(Uin * kernel)
        Uout = fft_object()
        Uout[:, :] = Uout / lamb / z * np.exp(2j * np.pi * z / lamb-np.pi / 2)
        Uout[:, :] = np.fft.ifftshift(Uout)
        Loy = Ny * lamb * z / 4 / Ly

    elif case == "planewave":
        # Backwards FFT
        inp2 = pyfftw.empty_aligned((Nx, Ny), dtype="complex128")
        out2 = pyfftw.empty_aligned((Nx, Ny), dtype="complex128")
        fft_object2 = pyfftw.FFTW(inp2, out2, axes=(0, 1),
                direction="FFTW_BACKWARD", flags=("FFTW_DESTROY_INPUT",))
        um, vm = Nx / 4 / Lx, Ny / 4 / Ly
        u = np.linspace(-um, um, Nx)
        v = np.linspace(-vm, vm, Ny)
        uu, vv = np.meshgrid(v, u)
        w2 = uu * uu + vv * vv
        # Free space transfer function
        # Shifting for convenience
        H = np.fft.fftshift(np.exp(-1j*np.pi*z*w2*lamb+\
                2j*np.pi*z/lamb))
        # Plane wave spectrum
        inp[:, :] = fftshift(Uin.astype(np.complex128))
        A = ifftshift(fft_object())
        # Multiplying the spectrum with the Free space transfer function
        inp2[:,:] = fftshift(A * H)
        # Getting the final amplitude
        Uout = ifftshift(fft_object2())
        # The dimensions of the window do not change
        Lox, Loy = Lx, Ly
    
    return Uout, (Lox, Loy)
