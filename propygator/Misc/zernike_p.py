#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from math import factorial  # Faster than scipy and numpy for int values

def zernike_p(rho, phi, n, m):
    """Calculation of a single term of Zernike polynomials. The function
    uses the theoretical (n,m) notation for the indices of the polynomial
    term. For a translation dictionary, see get_zernike_index.
    Input:
        - rho: Normalized radial component. If not normalized, 
               normalization will be applied beforehand.
        - phi: Azimuthal angle of the field.
        - n: Index determining the maximum order of the polynomial.
        - m: Index determining the number of zero crosses and the
             amount of periods the polynomial oscillates azimuthally.
    Returns:
        - R_nm*azimuth: Zernike polynomial of order (n, m)."""
    halfsub = (n - abs(m)) // 2
    halfsum = (n + abs(m)) // 2
    ncoef = halfsub  + 1
    # Azimuthal term
    azimuth = np.cos(m * phi) if m >= 0 else np.sin(m * phi)
    sgn = -1
    # Checking for array casting
    if type(rho) == np.ndarray:
        R = np.empty_like(rho)
    else:
        R = 0.

    # Calculation of the polynomial
    for s in range(ncoef):
        sgn *= -1
        R += sgn * factorial(n - s) // (factorial(s) *\
                factorial(halfsum - s) * factorial(halfsub - s)) *\
                        rho ** (n - 2 * s)
    return R * azimuth

def get_zernike_index(order="OSA"):
    """Return the (n, m) indices corresponding to one of three normalized
    representations of the Zernike coefficients."""
    if order == "OSA":
        indexs = {0:(0, 0), 1:(1, -1), 2:(1,1), 3:(2,-2), 4:(2, 0),
                5:(2, 2), 6:(3, -3), 7:(3, -1), 8:(3, 1), 9:(3, 3), 10:(4, -4),
                11:(4, -2), 12:(4, 0), 13:(4, 2), 14:(4, 4), 15:(5, -5),
                16:(5, -3), 17:(5, -1), 18:(5, 1), 19:(5, 3)}
    elif order == "Noll":
        # TODO: Change numeration in labels to reflect that Noll and 
        # Fringe start at 1
        indexs = {0:(0, 0), 2:(1, -1), 1:(1,1), 4:(2,-2), 3:(2, 0),
                5:(2, 2), 8:(3, -3), 6:(3, -1), 7:(3, 1), 9:(3, 3), 14:(4, -4),
                12:(4, -2), 10:(4, 0), 11:(4, 2), 13:(4, 4), 19:(5, 5),
                18:(5, -3), 16:(5, -1), 15:(5, 1), 17:(5, 3)}
    elif order == "Fringe":
        indexs = {0:(0, 0), 1:(1, 1), 2:(1, -1), 3:(2, 0), 4:(2, 2),
                5:(2, -2), 6:(3, 1), 7:(3, -1), 8:(4, 0), 9:(3, 3), 10:(3, -3),
                11:(4, 2), 12:(4, -2), 13:(5, 1), 14:(5, -1), 15:(6, 0), 
                16:(4, 4), 17:(4, -4), 18:(5, 3), 19:(5, -3)}
    else:
        raise ValueError("Index order not recognized")
    return indexs
