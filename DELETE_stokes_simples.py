#!/usr/bin/python3
"""
Calcula els Stokes segons la descripció de la Charo
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import os
    
def print_names(folders):
    for i, name in enumerate(folders):
        print(f"{i})\t{name}")

def get_folders():
    files = os.listdir()
    folders = []
    for name in files:
        if os.path.isdir(name):
            folders.append(name)

    folders.sort()

    selected = False
    while not selected:
        print_names(folders)
        args = input("> Type the numbers of the folders of interest separated "
                     "by an space (e.g. 0 2 5): ")
        try:
            folds = args.split(" ")
            folds = [int(i) for i in folds]
        except:
            print("Folders not recognized!")
            continue
        selected = True

    working_folders = []
    for i in folds:
        working_folders.append(folders[i])

    return working_folders

def get_z_component(Ex, Ey, p_size, lamb=520e-6, z=0):
    Ax = ifftshift(fft2(fftshift(Ex)))
    Ay = ifftshift(fft2(fftshift(Ey)))
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
        Ex = ifftshift(ifft2(fftshift(Ax)))
        Ey = ifftshift(ifft2(fftshift(Ay)))

    Ez = ifftshift(ifft2(fftshift(Az)))
    return Ex, Ey, Ez

def load_files(target):
    data_a = np.load(f"{target}/amplitudes.npz")
    data_p = np.load(f"{target}/phases.npz")

    Ax = data_a["Ax"]
    Ay = data_a["Ay"]
    p_size = data_a["p"]

    phi_x = data_p["phi_x"]
    phi_y = data_p["phi_y"]

    # plt.figure()
    # plt.imshow(np.angle(phi_x))
    # plt.colorbar()
    # plt.show()

    # return Ax*np.exp(1j*phi_x), Ay*np.exp(1j*phi_y), p_size
    return Ax * phi_x, Ay * phi_y, p_size

def compute_simple_stokes(Ex, Ey, Ez):
    ny, nx  = Ex.shape
    A = np.zeros((ny, nx, 3))
    B = np.zeros((ny, nx, 3))
    cross = np.zeros((ny, nx, 3))
    s = np.zeros((ny, nx, 4))

    # Construeixo els vectors
    A[:, :, 0] = np.real(Ex)
    A[:, :, 1] = np.real(Ey)
    A[:, :, 2] = np.real(Ez)

    B[:, :, 0] = np.imag(Ex)
    B[:, :, 1] = np.imag(Ey)
    B[:, :, 2] = np.imag(Ez)

    #
    A2 = np.sum(A*A, axis=-1)
    B2 = np.sum(B*B, axis=-1)
    cross[:, :, 0] = A[:, :, 1]*B[:, :, 2]-A[:, :, 2]*B[:, :, 1]
    cross[:, :, 1] = A[:, :, 2]*B[:, :, 0]-A[:, :, 0]*B[:, :, 2]
    cross[:, :, 2] = A[:, :, 0]*B[:, :, 1]-A[:, :, 1]*B[:, :, 0]
    
    # Computo l'angle alpha, tan 2a = 2 A · B / (|A|^2-|B|^2)
    alpha = .5*np.arctan2(2*np.sum(A*B, axis=-1), A2-B2)
    #alpha = np.arctan2(A2-B2, np.sum(A*B, axis=-1))

    # Vectors de Stokes per se
    s[:, :, 0] = A2+B2
    s[:, :, 1] = (A2-B2)/(np.cos(2*alpha))
    s[:, :, 2] = 0
    s[:, :, 3] = 2*np.sqrt(np.sum(cross*cross, axis=-1))

    #s[:, :, 1] /= s[:, :, 0]
    #s[:, :, 2] /= s[:, :, 0]
    #s[:, :, 3] /= s[:, :, 0]

    return s

def main():
    folders = get_folders()
    z = 0.0

    try:
        os.mkdir("Results")
    except:
        pass
    
    for kind in ("pRad", "pDex"):
        for folder in folders:
            print(f"Processing {folder}, {kind}")
            fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(8,7))
            Ex, Ey, p_size = load_files(f"{folder}/{kind}_retrieved")
            Ex, Ey, Ez = get_z_component(Ex, Ey, p_size, z=z)
            ny, nx = Ez.shape
            s = compute_simple_stokes(Ex, Ey, Ez)
            s /= s.max()
            L = p_size*ny/2*1e3

            ax[0, 0].imshow(s[:, :, 0], cmap="gray", extent=[-L, L, -L, L])
            ax[0, 0].set_title("$S_0$")
            ax[0, 1].imshow(s[:, :, 1], cmap="bwr", vmin=-1, vmax=1, extent=[-L, L, -L, L])
            ax[0, 1].set_title("$S_1$")
            ax[1, 0].imshow(s[:, :, 2], cmap="bwr", vmin=-1, vmax=1, extent=[-L, L, -L, L])
            ax[1, 0].set_title("$S_2$")
            pcm = ax[1, 1].imshow(s[:, :, 3], cmap="bwr", vmin=-1, vmax=1, extent=[-L, L, -L, L])
            ax[1, 1].set_title("$S_3$")

            for i in range(2):
                for j in range(2):
                    ax[i, j].set_xlabel("x ($\mathrm{\mu m}$)")
                    ax[i, j].set_ylabel("x ($\mathrm{\mu m}$)")

            # Barra de coloraines
            fig.colorbar(pcm, ax=ax[:, 1], shrink=1.0)

            # Desa les figures...
            fig.savefig(f"Results/{folder}_{kind}.png", bbox_inches="tight", dpi=200)

if __name__ == "__main__":
    main()
