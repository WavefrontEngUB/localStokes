"""
Definition of field objects to be used in the GUI
"""
from .HighAperture import compute_ref_sphere_field, compute_focal_field
from .PartialCoherence import compute_focal_field_pc
from .errors import DimensionError
from .Misc import zernike_p, get_zernike_index
import numpy as np
import pyfftw
from multiprocessing import cpu_count
pyfftw.config.NUM_THREADS = cpu_count()
pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
fftshift = np.fft.fftshift

class HighNAField(np.ndarray):
    """Object that represents the electric field distribution over the
    Entrance Pupil (E.P.) of a high Numerical Aperture (NA) optical system. The
    class derives from numpy arrays for convenience: calling it per se
    returns the field distribution over the E.P., while to obtain the
    field distribution over the focal region its method propagate() must
    be called. To build the possible aberrations of the system, one must call
    compute_aberration() after the creation of the object."""
    #TODO: Document the class
    def __new__(cls, input_array, f, NA, wdim, lamb, n=1.):
        # Creation of the class based on an already existing array
        in_shape = input_array.shape
        # Check if correct shape of array
        if len(in_shape) != 3:
            raise DimensionError("Input array must be of shape (nx, ny, 2).")
        if in_shape[-1] != 2:
            raise DimensionError("Input array must be of shape (nx, ny, 2).")
        Esp, kz, d = compute_ref_sphere_field(input_array,
                f, NA, *wdim, lamb, n=n)
        shape = Esp.shape
        obj = np.asarray(Esp).view(cls)
        # Adding the wavevectors
        obj.kz = kz
        #obj = np.asarray(input_array.astype(np.complex128)).view(cls)
        # Adding focal length of the optical system
        obj.f = f
        # Adding numerical aperture
        obj.NA = NA
        # Adding the size of the Entrance Pupil
        obj.d = d
        # Refractive index at the exit pupil
        obj.n = n
        # Adding the window half dimensions
        obj.wdim = wdim
        # Adding the wavelength of the radiation
        obj.lamb = lamb
        return obj

    def __array_finalize(self, obj):
        if obj is None: return
        self.wdim = getattr(obj, "wdim")
        self.f = getattr(obj, "f")
        self.NA = getattr(obj, "NA")
        self.n = getattr(obj, "n")
        self.lamb = getattr(obj, "lamb")
        self.d = getattr(obj, "d")
        self.kz = getattr(obj, "kz")

    def propagate(self, z, mu=None):
        """Propagate to a distance z from the focal plane of the optical
        system."""
        if type(mu) == np.ndarray:
            I = compute_focal_field_pc(self, mu, z, self.kz)
            return I
        else:
            E = compute_focal_field(self, z, self.kz)
            I = np.real(np.conj(E)*E)
            return I
    def construct_aberration(self, zernikes, order="OSA"):
        """Construct an aberration function to include in the
        Gaussian reference field. Aberrations are given as a set of
        Zernike Polynomials, ordered in one of three possible
        conventions:
            - OSA/ANSI
            - Noll
            - Fringe
        which must be selected to correctly describe the wave-front.
        The function takes, then, an array-like object (numpy array,
        list, tuple) and selects the non zero coefficients. Then, it
        calculates the aberration associated with each coefficient and
        multiplkies it with the Gaussian reference sphere field.
            -zernikes = dict(number : coefficient)
        """
        # TODO: Specify that zernikes must be a dictionary.
        # Defining the array to contain the pupil function
        nx, ny, _ = self.shape
        P = np.ones((nx, ny), dtype=np.complex128)
        pair_index = get_zernike_index(order=order)
        # Limits of the entrance pupil.
        Lx, Ly = self.wdim
        x = np.linspace(-Lx, Lx, nx)
        y = np.linspace(-Ly, Ly, ny)
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x*x+y*y)
        mask = r>=self.d
        phi = np.arctan2(y, x)
        r[mask] = 0
        r[:] = r/r.max()
        # Actual calculation of the Zernikes
        W = np.empty((nx, ny), dtype=np.float_)
        if type(zernikes) != dict:
            raise ValueError("Zernikes must be a dictionary.")
        ik = 2j*np.pi/self.lamb
        for key in zernikes:
            A = zernikes[key]
            n, m = pair_index[key]
            W[:] = A*zernike_p(r, phi, n, m)
            P[:] = P*np.exp(ik*W)
        # Finally, applying to each component
        for i in range(3):
            self[:, :, i] *= P
