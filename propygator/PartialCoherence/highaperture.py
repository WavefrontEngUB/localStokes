#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from multiprocessing import cpu_count
from ..HighAperture import ricwolf_int, compute_ref_sphere_field, compute_focal_field
import pyfftw
pyfftw.config.NUM_THREADS = cpu_count()
nthr = cpu_count()
pyfftw.config.PLANNER_EFFORT = "FFTW_MEASURE"

def ricwolf_int_pc(Ein, mu, f, NA, Lx, Ly, lamb, z=0, n=1, P=None):
    """Wrapper for a single computation of the focal field using
    electromagnetic diffraction theroy."""
    # TODO: Documentation
    Esp, kz, d = compute_ref_sphere_field(Ein, f, NA, Lx, Ly, lamb, n=n)
    nx, ny, _ = Esp.shape
    Ef = compute_focal_field_pc(Esp, mu, z=z, kz=kz, P=P)
    Lfx, Lfy = nx*lamb*f/Lx, ny*lamb*f/Ly
    return Ef, (Lfx, Lfy)

def compute_focal_field_pc(E_sp, mu, z, kz, P=None):
    """Compute the focal field provided E_sp and mu at a given z and with a 
    given set of aberration coefficients in the Zernike's basis"""
    # Compute the focal amplitude distribution for the completely coherent field.
    Ef = compute_focal_field(E_sp, z, kz, P=P)
    nx, ny, _ = Ef.shape
    # Autocorrelation of the FT of the intensity and correlation with
    # complex degree of coherence.
    inp = pyfftw.empty_aligned((nx, ny), dtype="complex128")
    out = pyfftw.empty_aligned((nx, ny), dtype="complex128")
    fft_obj = pyfftw.FFTW(inp, out, axes=(0, 1))

    inp2 = pyfftw.empty_aligned((nx, ny), dtype="complex128")
    out2 = pyfftw.empty_aligned((nx, ny), dtype="complex128")
    fft_obj2 = pyfftw.FFTW(inp2, out2, axes=(0, 1), direction="FFTW_BACKWARD")

    If = np.zeros((nx, ny, _), dtype=np.float64)
    mu_shift = np.fft.fftshift(mu)
    for i in range(3):
        inp[:, :] = np.conj(Ef[:, :, i])*Ef[:, :, i]
        fft_obj()

        # Convolution with the FT of the complex degree of coherence
        inp2[:, :] = mu_shift*out
        fft_obj2()
        If[:, :, i] = abs(out2)
    return If
