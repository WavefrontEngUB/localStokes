import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from propygator.HighAperture import debye_ricwol
from propygator.Misc import get_zernike_index, zernike_p

kind = "radial"

def ricardo_llop(E, NA, L, f, z=0):
    ny, nx, nc = E.shape
    Ef = np.zeros((ny, nx, 3), np.complex128)

    y, x = np.mgrid[-ny//2:ny//2, -nx//2:nx//2]
    y = y/y.max()*L
    x = x/x.max()*L

    r2 = x*x+y*y
    sinth2 = r2/f/f
    mask = sinth2 < NA*NA
    sinth = np.sqrt(sinth2)*(mask)

    costh = np.sqrt(1-sinth2, where=mask)
    costh *= mask
    sqcosth = np.sqrt(costh)
    phi = np.arctan2(y, x)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    # Multiplicació pels vectors unitaris...
    Ef[:, :, 0] = mask*E[:, :, 0]*(sinphi*sinphi+cosphi*cosphi*costh) +\
                  mask*E[:, :, 1]*sinphi*cosphi*(costh-1)
    Ef[:, :, 0][mask] /= sqcosth[mask]
    Ef[:, :, 1] = mask*E[:, :, 1]*(sinphi*sinphi*costh+cosphi*cosphi) +\
                  mask*E[:, :, 0]*sinphi*cosphi*(costh-1)*mask
    Ef[:, :, 1][mask] /= sqcosth[mask]
    Ef[:, :, 2] = -sinth*(E[:, :, 0]*cosphi + E[:, :, 1]*sinphi)
    Ef[:, :, 2][mask] /= sqcosth[mask]

    # Propagació
    if z != 0:
        H = np.exp(-2j*np.pi*z)*mask
        Ef[:, :, 0] *= H
        Ef[:, :, 1] *= H
        Ef[:, :, 2] *= H
    Ef[:, :, 0] = fftshift(fft2(Ef[:, :, 0]))
    Ef[:, :, 1] = fftshift(fft2(Ef[:, :, 1]))
    Ef[:, :, 2] = fftshift(fft2(Ef[:, :, 2]))
    return Ef

def get_z_component(Ex, Ey, p_size, lamb=520e-6, z=0):
    fft = lambda field: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(field)))
    ifft = lambda spectr: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(spectr)))

    Ax = fft(Ex)
    Ay = fft(Ey)
    ny, nx = Ex.shape

    y, x = np.mgrid[-ny//2:ny//2, -nx//2:nx//2]

    alpha = x/x.max()/p_size*lamb/(2*np.pi)
    beta = y/y.max()/p_size*lamb/(2*np.pi)

    gamma = np.sqrt(1+0j-alpha*alpha-beta*beta)
    Az = (alpha*Ax+beta*Ay)/gamma


    # Propaguem si s'escau
    if z > 0:
        H = np.exp(-2j*np.pi*gamma*z)
        H[alpha*alpha+beta*beta >= 1] = 0
        Az *= H
        Ax *= H
        Ay *= H
        Ex = ifft(Ax)
        Ey = ifft(Ay)

    Ez = ifft(Az)

    return Ex, Ey, Ez

if __name__ == "__main__":
    n = 512
    NA = 0.35
    lamb = 520e-6
    f = 5/lamb
    E = np.zeros((n, n, 2), dtype=np.complex128)
    y, x = np.mgrid[-n//2:n//2, -n//2:n//2]
    x = x+3
    y = y+3
    phi = np.arctan2(y, x)
    if kind == "radial":
        E[:, :, 0] = np.cos(phi)
        E[:, :, 1] = np.sin(phi)
    elif kind == "circular":
        E[:, :, 0] = 1
        E[:, :, 1] = -1j

    Lf = 16
    L = n*f/4/Lf

    # Càlcul Zernikes...
    rho = np.sqrt(x*x+y*y)/n*L/f
    rho[rho> NA] = 0
    rho/=rho.max()
    #
    coeffs = {0:0,
              1 :0e-2,
              2 :0e-2,
              3 :0e-2,
              5 :0e-2,
              6 :-0e-2,
              7 :0.0e-1,
              8 :0.0e-2,
              9 :0.0e-2,
              12:-0e-1,
             }
    idx = get_zernike_index("OSA")
    first = True
    for key in coeffs.keys():
        value = coeffs[key]
        n, m = idx[key]
        if first:
            R = value*zernike_p(rho, phi, n, m)
            first = False
        else:
            R += value*zernike_p(rho, phi, n, m)
    E[:, :, 0] *= np.exp(2j*np.pi*R)
    E[:, :, 1] *= np.exp(2j*np.pi*R)
    # Camp focal
    Ef = ricardo_llop(E, NA, L, f)

    # Stokes en coordenades estranyes
    import import_ipynb
    from Report import compute_3D_stokes
    #Ef[:, :, 0] = np.roll(Ef[:, :, 0], (4, -4), axis=(0, 1))
    s = compute_3D_stokes(Ef[:, :, 0], Ef[:, :, 1], Ef[:, :, 2])
    s /= s.max()

    I = np.real(np.conj(Ef)*Ef)
    fig, ax = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle(f"Irradiances in cartesian basis, {kind}")

    extent = (-Lf, Lf, -Lf, Lf)
    extent = [l*lamb*1000 for l in extent]
    cmap="gray"
    ax[0, 0].imshow(I[:, :, 0], cmap=cmap, extent=extent)
    ax[0, 0].set_title("$I_x$")
    ax[0, 1].imshow(I[:, :, 1], cmap=cmap, extent=extent)
    ax[0, 1].set_title("$I_y$")
    ax[1, 0].imshow(I[:, :, 2], cmap=cmap, extent=extent)
    ax[1, 0].set_title("$I_z$")
    ax[1, 1].imshow(np.sum(I, axis=-1), cmap=cmap, extent=extent)
    ax[1, 1].set_title("$I_t$")

    fig2, ax2 = plt.subplots(2, 2, constrained_layout=True, figsize=(8,8))
    fig2.suptitle(f"Stokes in strange basis, {kind}")
    ax2[0, 0].imshow(s[:, :, 0], cmap=cmap, extent=extent)
    ax2[0, 0].set_title("$S_0$")
    ax2[0, 1].imshow(s[:, :, 1], cmap="bwr", extent=extent, vmin=-1, vmax=1)
    ax2[0, 1].set_title("$s_1$")
    ax2[1, 0].imshow(s[:, :, 2], cmap="bwr", extent=extent, vmin=-1, vmax=1)
    ax2[1, 0].set_title("$s_2$")
    ax2[1, 1].imshow(s[:, :, 3], cmap="bwr", extent=extent, vmin=-1, vmax=1)
    ax2[1, 1].set_title("$s_3$")

    ax2[0, 0].set_xlabel("x ($\mathrm{\mu m}$)")
    ax2[0, 0].set_ylabel("y ($\mathrm{\mu m}$)")
    ax2[0, 1].set_xlabel("x ($\mathrm{\mu m}$)")
    ax2[0, 1].set_ylabel("y ($\mathrm{\mu m}$)")
    ax2[1, 0].set_xlabel("x ($\mathrm{\mu m}$)")
    ax2[1, 0].set_ylabel("y ($\mathrm{\mu m}$)")
    ax2[1, 1].set_xlabel("x ($\mathrm{\mu m}$)")
    ax2[1, 1].set_ylabel("y ($\mathrm{\mu m}$)")

    fig2.savefig(f"{kind}.png", bbox_inches="tight", dpi=200)
    plt.show()
