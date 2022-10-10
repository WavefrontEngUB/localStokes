import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from stokes_simples import compute_simple_stokes

kind = "radial"  # "radial" or "circular"


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

if __name__ == "__main__":
    n = 512
    NA = 0.45
    lamb = 520e-6
    f = 5/lamb
    E = np.zeros((n, n, 2), dtype=np.complex128)
    y, x = np.mgrid[-n//2:n//2, -n//2:n//2]
    phi = np.arctan2(y, x)
    if kind == "radial":
        E[:, :, 0] = np.cos(phi)
        E[:, :, 1] = np.sin(phi)
    elif kind == "circular":
        E[:, :, 0] = 1
        E[:, :, 1] = 1j

    Lf = 16
    L = n*f/4/Lf

    # Camp focal
    Ef = ricardo_llop(E, NA, L, f)

    # Stokes en coordenades estranyes
    s = compute_simple_stokes(Ef[:, :, 0], Ef[:, :, 1], Ef[:, :, 2])

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

    fig2.savefig(f"{kind}_{NA}.png", bbox_inches="tight", dpi=200)
    plt.show()
