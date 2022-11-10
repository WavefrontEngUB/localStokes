#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from multiprocessing import cpu_count
import pyfftw
pyfftw.config.NUM_THREADS = cpu_count()
nthr = cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"

def freespace_pc(Uin, mu, Lx, Ly, z, lamb):
    """Partially coherent free space propagation. The follwing approximations have been
    taken into account:
        1. Gauss-Schell approximation to the Cross Spectral Density Function:
            W(x1, y1, x2, y2) = S(x1, y1)^(1/2) S(x2, y2)^(1/2) mu(x2-x1, y2-y1)
        2. Paraxial approximation (Fresnel regime).
    The propagation is performed according to
        f = S(x1, y1)^(1/2) exp[i pi/lamb/z*(x^2+x^2)]

        I = FT[(f^star * f) mu]
    where * denotes the convolution operator.

    To get the correct sampling under all situations, when the propagation distance
    z is short compared to the dimensions of the sampling window and the wavelength,
    the following propagation law is used:
        I = M(r/lamb/z) * [U^star(r; z) U(r; z)]
    where M() is the FT of the spectral degree of coherence and U(r; z) is the wave
    amplitude propagated to the plane z.

    The function takes a complex amplitude and the spectral degree of coherence
    as inputs and outputs the intensity at the plane situated at z from the input.
    Input:
        - Uin: Optical disturbance or electric field component.
        - mu: Spectral degree of coherence in the input plane.
        - Lx, Ly: half widths of the sampling window.
        - z: distance between the input and output planes.
        - lamb: wavelength of the light.
    Output:
        - I: Intensity at the output plane
        - (Lox, Loy): Half widths of the output sampling window.
    """
    try:
        (Nx, Ny) = Uin.shape
    except:
        DimensionError("Uin must be a 2D array")
    if Uin.shape != mu.shape:
        raise DimensionError("Uin and mu must have the same shape.")

    # Selecting the correct regime
    Lox = Nx * lamb * z / 4 / Lx  # If squared sampling, equal for y
    case = "fresnel" if  Lox >= Lx else "planewave"


    # Preparing the transformations
    inp = pyfftw.empty_aligned((Nx, Ny), dtype="complex128")
    out = pyfftw.empty_aligned((Nx, Ny), dtype="complex128")
    fft_obj = pyfftw.FFTW(inp, out, axes=(0, 1))

    inp2 = pyfftw.empty_aligned((Nx, Ny), dtype="complex128")
    out2 = pyfftw.empty_aligned((Nx, Ny), dtype="complex128")
    fft_obj2 = pyfftw.FFTW(inp2, out2, axes=(0, 1), direction="FFTW_BACKWARD")

    # Propagating using the proper regime
    if  case == "fresnel":
        ct = (1/lamb/z)
        ct *= ct
        # Preparing the inputs
        x = np.linspace(-Lx, Lx, Nx)
        y = np.linspace(-Ly, Ly, Ny)
        xx, yy = np.meshgrid(y, x)
        f = Uin * np.exp(1j*np.pi*(xx*xx + yy*yy)/lamb/z)
    
        # Autocorrelation of the input
        inp[:, :] = f
        fft_obj()
        inp2[:, :] = np.conj(out)*out
        fft_obj2()
        # Fourier transform of the result
        inp[:, :] = out2 * np.fft.fftshift(mu)
        fft_obj()
        Iout = abs(np.fft.fftshift(out))*ct
        Loy = Ny * lamb * z / 4 / Ly  # If squared sampling, equal for y

    elif case == "planewave":
        um, vm = Nx / 4 / Lx, Ny / 4 / Ly
        Lox, Loy = Lx, Ly
        u = np.linspace(-um, um, Nx)
        v = np.linspace(-vm, vm, Ny)
        uu, vv = np.meshgrid(v, u)
        w2 = uu * uu + vv * vv

        
        # Calculation of F = h * Uin
        # Free space transfer function
        #H = np.fft.fftshift(np.exp((-1j * np.pi * z * w2)*lamb))
        #sc = -1j*z*lamb*.25/np.pi
        sc = -1j*np.pi*z*lamb
        #sc = -1j * np.pi*z*lamb
        H = np.exp(sc*w2)
        
        # Calculation of the amplitude at the specified z distance. Identical
        # with ../Propagation/propagation.py freespace.
        inp[:, :] = Uin.astype(np.complex128)
        fft_obj()
        inp2[:, :] = out*np.fft.fftshift(H)
        fft_obj2()
        
        # Convolution with the FT of the spectral degree of coherence
        inp[:, :] = np.real(np.conj(out2)*out2)
        fft_obj()
        inp2[:, :] = out * np.fft.fftshift(mu)
        fft_obj2()
        Iout = np.abs(out2)

    return Iout, (Lox, Loy)
