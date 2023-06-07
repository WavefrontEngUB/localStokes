import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from propygator.HighAperture import debye_ricwol
from propygator.Misc import get_zernike_index, zernike_p

fft = lambda field: fftshift(fft2(ifftshift(field)))
ifft = lambda spectr: ifftshift(ifft2(fftshift(spectr)))

kind = "radial"

def ricardo_llop(E, NA, L, f, z=0):
    ny, nx, nc = E.shape
    Ef = np.zeros((ny, nx, 3), np.complex128)

    _, _, coords = get_EP_coords(NA, 520e-3, nx)

    # y, x = coords["y"], coords["x"]
    # r2 = coords["r2"]
    mask = coords["mask"]
    sinth = coords["sinth"]
    costh = coords["costh"]
    sqcosth = coords["sqcosth"]
    sinphi = coords["sinphi"]
    cosphi = coords["cosphi"]

    # y, x = np.mgrid[-ny//2:ny//2, -nx//2:nx//2]
    # y = y/y.max()*L
    # x = x/x.max()*L
    #
    # r2 = x*x+y*y
    # sinth2 = r2/f/f
    # mask = sinth2 < NA*NA
    # sinth = np.sqrt(sinth2)*(mask)
    #
    # costh = np.sqrt(1-sinth2, where=mask)
    # costh *= mask
    # sqcosth = np.sqrt(costh)
    # phi = np.arctan2(y, x)
    # sinphi = np.sin(phi)
    # cosphi = np.cos(phi)

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
    Ef[:, :, 0] = fft(Ef[:, :, 0])
    Ef[:, :, 1] = fft(Ef[:, :, 1])
    Ef[:, :, 2] = fft(Ef[:, :, 2])
    return Ef

def get_z_component(Ex, Ey, p_size, lamb=520e-3, z=0):
    Ax = fft(Ex)
    Ay = fft(Ey)
    ny, nx = Ex.shape

    y, x = np.mgrid[-ny//2:ny//2, -nx//2:nx//2]

    alpha = x/x.max()/p_size*lamb*0.5
    beta = y/y.max()/p_size*lamb*0.5

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


def get_theoretical_field(kind, sigma=2, NA=0.7, lamb=520e-3, n=256, verbose=0, mario=False, m_topo=0):
    """ kind = "radial", "circular" or "lineal"
    """
    L, f, coords = get_EP_coords(NA, lamb, n)

    E = np.zeros((n, n, 2), dtype=np.complex128)

    # profile
    g = np.zeros((n, n), dtype=np.complex128)
    g += np.exp( -sigma/2 * (coords["costh"]-coords["alpha_bar"])**2/(1-coords["alpha_0"])**2 )
    g /= coords["sqcosth"] * (1 - coords["costh"]) * coords["mask"]
    if kind == "radial":
        g *= coords["sinth"]
    elif kind == "takoma":
        # Gaussina profile with topological charge
        g = np.ones((n, n), dtype=np.complex128)
        g *= np.exp(-coords["r2"] / (2 * sigma ** 2))
        g *= np.exp(1j * coords["phi"] * m_topo)
        g *= coords["sinth"]
    g *= coords["mask"]

    # polarization
    if kind == "radial" or kind == "takoma":
        E[:, :, 0] = coords["cosphi"]
        E[:, :, 1] = coords["sinphi"]
    elif kind == "circular":
        E[:, :, 0] = 1
        E[:, :, 1] = 1j
    elif kind == "linear":
        E[:, :, 0] = 1
        E[:, :, 1] = 0

    E *= np.repeat(g[:, :, np.newaxis], 2, axis=2)
    np.nan_to_num(E, copy=False)

    plot_raw_trans_field(E[:,:,0], E[:,:,1], "*** EP field ***") if verbose>1 else None

    # Camp focal
    return ricardo_llop(E, NA, L, f)


def get_EP_coords(NA, lamb, n):
    f = 5 / lamb
    Lf = 16
    L = n * f / 4 / Lf

    y, x = np.mgrid[-n // 2:n // 2, -n // 2:n // 2]
    y = y / y.max() * L
    x = x / x.max() * L

    r2 = x * x + y * y
    sinth2 = r2 / f / f
    mask = sinth2 < NA * NA

    sinth = np.sqrt(sinth2) * (mask)
    costh = np.sqrt(1 - sinth2, where=mask)
    costh *= mask
    sqcosth = np.sqrt(costh)
    theta = np.arcsin(sinth) * mask

    theta_0 = np.arcsin(NA)
    alpha_0 = np.cos(theta_0)
    alpha_bar = (alpha_0 + 1) * 0.5

    phi = np.arctan2(y, x)
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)

    return L, f, {"x": x, "y": y, "r2": r2, "mask": mask,
                  "theta": theta, "sinth": sinth, "costh": costh, "sqcosth": sqcosth,
                  "phi": phi, "sinphi": sinphi, "cosphi": cosphi,
                  "theta_0":alpha_0, "alpha_0": alpha_0, "alpha_bar": alpha_bar}

def print_fig(msg, fig_num):
    fig_num += 1
    print(f"Figure {fig_num}: {msg}")
    return fig_num


def plot_raw_data(retriever_obj, label='', fig_num=0, cmap="gray"):
    irradiances = [retriever_obj.cropped[0][pol] for pol in range(6)]
    maximum = max([np.max(irr) for irr in irradiances])

    pol_keys=["$I_{0}$", "$I_{45}$", "$I_{90}$", "$I_{135}$", "$I_{Lev}$", "$I_{Dex}$"]

    fig = plt.figure(figsize=(20, 30))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(2, 3),
                    axes_pad=0.6,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.2
                    )

    for idx, ax in enumerate(axs):
        irr = irradiances[idx]
        im = ax.imshow(irr/maximum, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(pol_keys[idx])

    cbar = plt.colorbar(im, cax=axs.cbar_axes[0], ticks=[0, 1])
    [t.set_fontsize(20) for t in cbar.ax.get_yticklabels()]
    plt.show()
    title = f"({label} Pol.) The raw irradiances captured by the camera."
    return print_fig(title, fig_num)

def plot_raw_trans_field(Ex, Ey, label='', fig_num=0, cmap_abs='jet', cmap_ph='hsv'):
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(10,10))
    ax[0, 0].imshow(np.abs(Ex), cmap=cmap_abs)
    ax[0, 0].set_title(r"$|Ex|$")
    ax[1, 0].imshow(np.abs(Ey), cmap=cmap_abs)
    ax[1, 0].set_title(r"$|Ey|$")
    ax[0, 1].imshow(np.angle(Ex), cmap=cmap_ph)
    ax[0, 1].set_title(r"$\phi_x$")
    ax[1, 1].imshow(np.angle(Ey), cmap=cmap_ph)
    ax[1, 1].set_title(r"$\phi_y$")
    plt.show()
    title = f"({label} Pol.) The raw data of the retrieved transversal component."
    return print_fig(title, fig_num)


def plot_raw_long_field(Ez, label='', fig_num=0, cmap_abs='jet', cmap_ph='hsv'):
    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(10,20))
    ax[0].imshow(np.abs(Ez), cmap=cmap_abs)
    ax[0].set_title(r"$|Ez|$")
    ax[1].imshow(np.angle(Ez), cmap=cmap_ph)
    ax[1].set_title(r"$\phi_z$")
    plt.show()
    title = f"({label} Pol.) The raw data of the retrieved longitudinal component."
    return print_fig(title, fig_num)

def plot_fields(Ex, Ey, Ez, trans_stokes=None, fig_num=0, trim=None, label="", verbose=1,
                pixel_size=1, lamb=520e-3, ticks_step=1):
    """

    :param Ex:
    :param Ey:
    :param Ez:
    :param fig_num:
    :param trim:
    :param label:
    :param verbose:
    :param pixel_size:
    :param lamb:
    :param ticks_step: in lambdas
    :return:
    """
    ntrim = -trim if trim else None
    lims = Ex[trim:ntrim, trim:ntrim].shape

    extension = lims[0] * pixel_size / lamb  # window size in microns
    half_side = extension / 2
    extent = [-half_side, half_side, -half_side, half_side]

    ticks = [0]
    for tt in range(int(extension / ticks_step / 2)):
        tt1 = tt + 1
        ticks.append(tt1 * ticks_step)
        ticks.insert(0, -tt1 * ticks_step)

    fig = plt.figure(figsize=(20, 40))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(2, 5),
                    axes_pad=0.6,
                    cbar_location="right",
                    cbar_mode="edge",
                    cbar_size="5%",
                    cbar_pad=0.2
                    )

    ph_range = lambda delta: np.angle(np.exp(1j * delta))  # from [0, 2pi] to [-pi, pi]

    Ax = np.abs(Ex[trim:ntrim, trim:ntrim])
    Ay = np.abs(Ey[trim:ntrim, trim:ntrim])
    Az = np.abs(Ez[trim:ntrim, trim:ntrim])
    phx = np.angle(Ex[trim:ntrim, trim:ntrim])  # range [-pi, pi]
    phy = np.angle(Ey[trim:ntrim, trim:ntrim])  # range [-pi, pi]
    phz = np.angle(Ez[trim:ntrim, trim:ntrim])  # range [-pi, pi]

    IT = np.sqrt(Ax ** 2 + Ay ** 2 + Az ** 2)
    Itrans = np.sqrt(Ax ** 2 + Ay ** 2)
    maxIT = np.max(IT)

    ph = ph_range(phy-phx)  # range [-pi, pi]

    exp_phase_shift = (np.arctan2(trans_stokes[3], trans_stokes[2]) if trans_stokes else
                       np.angle(Ey))  # range [-pi, pi]
    exp_phase_shift = exp_phase_shift[trim:ntrim, trim:ntrim]
    ph_error = ph_range(ph - exp_phase_shift) # range [-pi, pi]

    my_greens = np.zeros((256, 4))
    my_greens[:, 1] = np.linspace(0, 1, 256)
    my_greens[:, 3] = 1
    cmap_amps = ListedColormap(my_greens)
    cmap_phs = 'twilight_shifted'

    normalize = lambda a: a / a.max()  # (a-a.min())/(a.max()-a.min())

    if verbose > 1:
        get_title = lambda name, im: (fr'${name} ; '
                                      fr'vmin={im.min():.2g} ; '
                                      fr'vmax={im.max():.2g}$')
        fs = 14
    else:
        bra = '{'
        ket = '}'
        get_title = lambda name, im: (f'${name}$  $_{bra}[max={(im.max()):0.2f}]{ket}$')
        fs = 20

    for idx, ax in enumerate(axs):
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \, / \, \lambda$', fontsize=20)
        ax.set_ylabel(r'$y \, / \, \lambda$', fontsize=20)

        if idx == 0:
            im = ax.imshow(normalize(Ax), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title('|E_x|', Ax/maxIT), fontsize=fs)
        elif idx == 1:
            im = ax.imshow(normalize(Ay), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title('|E_y|', Ay/maxIT), fontsize=fs)
        elif idx == 2:
            im = ax.imshow(normalize(Itrans), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title(r'\sqrt{|E_x|^2+|E_y|^2}', Itrans/maxIT), fontsize=fs)
        elif idx == 3:
            im = ax.imshow(normalize(Az), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title('|E_z|', Az/maxIT), fontsize=fs)
        elif idx == 4:
            im = ax.imshow(normalize(IT), cmap=cmap_amps,
                           vmin=0, vmax=1, extent=extent)
            ax.set_title(get_title(r'\sqrt{I_T}', IT/maxIT), fontsize=fs)
        elif idx == 5:
            im = ax.imshow(phx, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\delta_x$', fontsize=fs)
        elif idx == 6:
            im = ax.imshow(phy, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\delta_y$', fontsize=fs)
        elif idx == 7:
            im = ax.imshow(ph, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\tilde{\delta}=\delta_y-\delta_x$', fontsize=fs)
        elif idx == 8:
            im = ax.imshow(phz, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\delta_z$', fontsize=fs)
        elif idx == 9 and trans_stokes:
            im = ax.imshow(ph_error, cmap=cmap_phs,
                           vmin=-np.pi, vmax=np.pi, extent=extent)
            ax.set_title(r'$\epsilon_\delta=\tilde{\delta}-\delta_{exp}$', fontsize=fs)


        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=20)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=20)
        if idx == 5:
            cbar = axs.cbar_axes[1].colorbar(im,  # FIXME: The line below is not working
            # cbar = fig.colorbar(im, ax=ax, cax=axs.cbar_axes[idx], orientation='vertical', shrink=0.5,
                                             ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2,
                                                    np.pi])
            cbar.ax.set_yticklabels(
                [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'], fontsize=20)
        elif idx == 2:
            cbar = fig.colorbar(im, ax=ax, cax=axs.cbar_axes[0], orientation='vertical',
                                shrink=0.5,
                                ticks=[0, 1])
            cbar.ax.set_yticklabels([r'0', r'max'], fontsize=20)

    plt.show()
    return print_fig(f"({label} Pol.) Field in the focal plane.", fig_num)

def plot_polarization_ellipse(Ex, Ey, Ez, n, m, trim=None, ticks_step=1,pixel_size=1,lamb=520e-3,
                             label='', fig_num=1, verbose=1, ax=None, format='-k'):
    """ Plot nxn ellipses of polarization using m points,
        according to the transversal components of the field.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \, / \, \lambda$', fontsize=20)
        ax.set_ylabel(r'$y \, / \, \lambda$', fontsize=20)

    ntrim = -trim if trim else None
    lims = Ex[trim:ntrim, trim:ntrim].shape

    extension = lims[0] * pixel_size / lamb  # window size in microns
    half_side = extension / 2
    extent = [-half_side, half_side, -half_side, half_side]

    ticks = [0]
    for tt in range(int(extension / ticks_step / 2)):
        tt1 = tt + 1
        ticks.append(tt1 * ticks_step)
        ticks.insert(0, -tt1 * ticks_step)

    Ax = np.abs(Ex[trim:ntrim, trim:ntrim])
    Ay = np.abs(Ey[trim:ntrim, trim:ntrim])
    size = Ax.shape[0]
    Ax /= Ax.max()
    Ay /= Ay.max()

    phx = np.angle(Ex[trim:ntrim, trim:ntrim])
    phy = np.angle(Ey[trim:ntrim, trim:ntrim])

    for i in range(n):
        for j in range(n):
            """ plot nxn points equaly distributed in a square of sizexsize """
            x = (i + 0.5) * size / n
            y = (j + 0.5) * size / n
            x1 = []
            y1 = []
            for k in range(m):
                """ plot m points in each ellipse """
                theta = k * 2 * np.pi / m
                ellipse_x = np.cos(theta + phx[int(x), int(y)]) * Ax[int(x), int(y)]
                ellipse_y = np.cos(theta + phy[int(x), int(y)]) * Ay[int(x), int(y)]

                ellipse_step = size / n / 2.3

                ellipse_x *= ellipse_step
                ellipse_y *= ellipse_step

                xx = x + ellipse_x
                yy = y + ellipse_y

                x1.append(xx)
                y1.append(yy)

            ax.plot(x1+[x1[0]], y1+[y1[0]], format, markersize=1, linewidth=1,
                    label=label if i == 0 and j == 0 else None)

    # ax.set_xticks([t/pixel_size * lamb + size/2 for t in ticks])
    # ax.set_xticklabels(ticks, fontsize=20)
    # ax.set_yticks([t/pixel_size * lamb + size/2 for t in ticks])
    # ax.set_yticklabels(ticks, fontsize=20)

    return ax, ticks, [t/pixel_size * lamb + size/2 for t in ticks]

def plot_trans_stokes(S0, S1, S2, S3, label, trim=None,
                      pixel_size=1, lamb=520e-3, fig_num=0, verbose=1):
    """ Plot the Stokes parameters of the transmitted field.
    """
    ntrim = -trim if trim else None
    lims = S0[trim:ntrim, trim:ntrim].shape

    ticks_step = 1  # lambda

    extension = lims[0] * pixel_size / lamb  # window size in microns
    half_side = extension / 2
    extent = [-half_side, half_side, -half_side, half_side]

    ticks = [0]
    for tt in range(int(extension / ticks_step / 2)):
        tt1 = tt + 1
        ticks.append(tt1 * ticks_step)
        ticks.insert(0, -tt1 * ticks_step)

    fig = plt.figure(figsize=(20, 40))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(1, 4),
                    axes_pad=0.6,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.2
                    )

    # Ax = np.abs(Ex[trim:ntrim, trim:ntrim])
    # Ay = np.abs(Ey[trim:ntrim, trim:ntrim])
    # phx = np.angle(Ex[trim:ntrim, trim:ntrim])
    # phy = np.angle(Ey[trim:ntrim, trim:ntrim])
    #
    # S0 = Ax ** 2 + Ay ** 2
    # S1 = Ax ** 2 - Ay ** 2
    # S2 = 2 * Ax * Ay * np.cos(phy - phx)
    # S3 = 2 * Ax * Ay * np.sin(phy - phx)

    stokes = [S0[trim:ntrim, trim:ntrim]/S0.max(), S1[trim:ntrim, trim:ntrim]/S0.max(),
              S2[trim:ntrim, trim:ntrim]/S0.max(), S3[trim:ntrim, trim:ntrim]/S0.max()]

    if verbose > 1:
        get_title = lambda name, im: (f'${name} ; '
                                      f'vmin={im.min():.2g} ; '
                                      f'vmax={im.max():.2g}$')
        fs = 14
    else:
        get_title = lambda name, im: (f'${name}$')
        fs = 20

    cmap = 'seismic'

    for idx, ax in enumerate(axs):
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \, / \, \lambda$', fontsize=20)
        ax.set_ylabel(r'$y \, / \, \lambda$', fontsize=20)

        im = ax.imshow(stokes[idx], vmin=-1, vmax=1,
                       cmap=cmap, extent=extent)
        ax.set_title(get_title(f'S_{idx}', stokes[idx]), fontsize=fs)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=20)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=20)

    cbar = plt.colorbar(im, cax=axs.cbar_axes[0], ticks=[-1, 0, 1])
    [t.set_fontsize(20) for t in cbar.ax.get_yticklabels()]
    plt.show()

    return print_fig(f"({label} Pol.) Experimental transversal Stokes "
                     f"in the focal plane.", fig_num)


def plot_3D_stokes(experimental_s, theoric_s, label, pixel_size=1, lamb=520e-3,
                   trim=None, fig_num=0):
    ntrim = -trim if trim else None
    lims = experimental_s[0][trim:ntrim, trim:ntrim].shape

    ticks_step = 1  # lambda

    extension = lims[0] * pixel_size / lamb  # window size in microns
    half_side = extension / 2
    extent = [-half_side, half_side, -half_side, half_side]

    ticks = [0]
    for tt in range(int(extension / ticks_step / 2)):
        tt1 = tt + 1
        ticks.append(tt1 * ticks_step)
        ticks.insert(0, -tt1 * ticks_step)

    fig = plt.figure(figsize=(20, 30))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(2, 3),
                    axes_pad=0.6,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.2
                    )
    S_tilde = r'\tilde{S}'
    titles = [f'${S_tilde}_{i}$' for i in range(4)]
    titles.insert(2, f'${S_tilde}'+r'_1^{num}$')
    titles.insert(5, f'${S_tilde}'+r'_3^{num}$')

    stokes = [(experimental_s[:, :, i]) for i in range(4)]
    stokes.insert(2, (theoric_s[:, :, 1]))
    stokes.insert(5, (theoric_s[:, :, 3]))
    stokes[3] = np.zeros_like(experimental_s[:, :, 2])  # all ZEROS generates NaN

    for idx, ax in enumerate(axs):
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \, / \, \lambda$', fontsize=20)
        ax.set_ylabel(r'$y \, / \, \lambda$', fontsize=20)

        im = ax.imshow(stokes[idx][trim:ntrim, trim:ntrim],
                       cmap='Reds', vmin=0, vmax=1, extent=extent)  #
        ax.set_title(titles[idx], fontsize=20)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=20)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=20)

    cbar = plt.colorbar(im, cax=axs.cbar_axes[0], ticks=[0, 1])
    [t.set_fontsize(20) for t in cbar.ax.get_yticklabels()]
    plt.show()

    return print_fig(f"({label} Pol.) 3D Stokes in the PQ-basis and "
                     f"the numerical calculation for comparison purposes.", fig_num)


def plot_paper_fig(trans_stokes, local_stokes, local_stokes_num, ticks_step = 1,
                   label='', pixel_size=1, lamb=520e-3, fig_num=0, trim=100):
    fig = plt.figure(figsize=(10,7.5), layout='tight')
    subfigs = fig.subfigures(2, 1, height_ratios=(1,2.75))

    cmap = 'seismic'
    ntrim = -trim if trim else None
    lims = local_stokes[0][trim:ntrim, trim:ntrim].shape

    extension = lims[0] * pixel_size / lamb  # window size in microns
    half_side = extension / 2
    extent = [-half_side, half_side, -half_side, half_side]

    ticks = [0]
    for tt in range(int(extension / ticks_step / 2)):
        tt1 = tt + 1
        ticks.append(tt1 * ticks_step)
        ticks.insert(0, -tt1 * ticks_step)

    axs_t = ImageGrid(subfigs[0], 111,
                      nrows_ncols=(1, 4),
                      axes_pad=0.1
                      )

    axs_l = ImageGrid(subfigs[1], 111,
                      nrows_ncols=(2, 3),
                      axes_pad=0.1
                      )

    for idx, ax in enumerate(axs_t):
        ax.set_aspect('equal')
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(r'$x \, / \, \lambda$', fontsize=20)
        ax.set_ylabel(r'$y \, / \, \lambda$', fontsize=20)

        im = ax.imshow(trans_stokes[idx][trim:ntrim,trim:ntrim]/trans_stokes[0].max(), vmin=-1, vmax=1,
                       cmap=cmap, extent=extent)
        ax.annotate(f'$S_{idx}$', (0.05,0.05), xycoords='axes fraction', fontsize=18)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=16)
        ax.xaxis.tick_top()
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=16)

    s_indices = [0, 1, 3]*2
    for idx, ax in enumerate(axs_l):
        s_label = r'\tilde{S}' if idx < 3 else r'\tilde{S}^{num}'

        stokes_l = local_stokes if idx < 3 else local_stokes_num
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \, / \, \lambda$', fontsize=20)
        ax.set_ylabel(r'$y \, / \, \lambda$', fontsize=20)

        im = ax.imshow(stokes_l[trim:ntrim,trim:ntrim,s_indices[idx]], vmin=-1, vmax=1,
                       cmap=cmap, extent=extent)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=16)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=16)
        ax.annotate(f'${s_label}_{s_indices[idx]}$', (0.05,0.05), xycoords='axes fraction', fontsize=18)

    fig.subplots_adjust(right=0.75)
    cbar_ax = fig.add_axes([1, 0.0122, 0.03, 0.98])
    cbar = plt.colorbar(im, cax=cbar_ax, ticks=[-1, 0, 1])
    [t.set_fontsize(20) for t in cbar.ax.get_yticklabels()]

    plt.show()

    return print_fig(f"({label} Pol.) Main figure in the paper. "
                     f"(top) Transversal Stokes images, "
                     f"(center) Experimental local Stokes images in the PQ-basis, and "
                     f"(bottom) Numerical calculation of local Stokes images "
                     f"for comparison purposes.", fig_num)


def plot_takoma_fig(trans_stokes, local_stokes, label, pixel_size=1, lamb=520e-3,
                    trim=None, fig_num=0):
    ntrim = -trim if trim else None
    lims = trans_stokes[0][trim:ntrim, trim:ntrim].shape

    ticks_step = 1  # lambda

    extension = lims[0] * pixel_size / lamb  # window size in microns
    half_side = extension / 2
    extent = [-half_side, half_side, -half_side, half_side]

    ticks = [0]
    for tt in range(int(extension / ticks_step / 2)):
        tt1 = tt + 1
        ticks.append(tt1 * ticks_step)
        ticks.insert(0, -tt1 * ticks_step)

    fig = plt.figure(figsize=(20, 30))
    axs = ImageGrid(fig, 111,
                    nrows_ncols=(2, 4),
                    axes_pad=0.6,
                    cbar_location="right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.2
                    )
    S_tilde = r'\tilde{S}'
    titles = [f'$S_{i}$' for i in range(4)]
    titles += [f'${S_tilde}_{i}$' for i in range(4)]


    for idx, ax in enumerate(axs):
        if idx < 4:
            stokes = trans_stokes
        else:
            stokes = local_stokes
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x \, / \, \lambda$', fontsize=20)
        ax.set_ylabel(r'$y \, / \, \lambda$', fontsize=20)

        im = ax.imshow(stokes[trim:ntrim, trim:ntrim, idx%4]/stokes[:,:,0].max(),
                       cmap='seismic', vmin=-1, vmax=1, extent=extent)  #
        ax.set_title(titles[idx], fontsize=20)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=20)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=20)

    cbar = plt.colorbar(im, cax=axs.cbar_axes[0], ticks=[0, 1])
    [t.set_fontsize(20) for t in cbar.ax.get_yticklabels()]
    plt.show()

    return print_fig(f"({label}) Comparation of Transversal and Local "
                     f"Stokes parameters.", fig_num)


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
