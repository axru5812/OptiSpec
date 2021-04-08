import numpy as np
from astropy import constants

def gauss(x, sigma, mu):
    """ Simple gaussian function that returns a normalized gaussian
    Parameters
    ----------
    x : array
        wavelength array
    sigma : float
        standard deviation
    mu : float
        mean
    """
    normalization = 1. / (sigma * np.sqrt(2 * np.pi))
    gaussian = np.e ** (-(1 / 2) * ((x - mu) / sigma) ** 2)
    return normalization * gaussian


def add_line(x, rest_center_wl, redshift, fwhm_kms, flux):
    """ Function that adds a gaussian line to an existing spectrum
    Parameters
    ----------
    x : array
        wavelength array
    rest_center_wl : float
        Central wavelength of the line
    redshift : float
        Redshift of the spectrum
    fwhm_kms : float
        FWHM of the line in kilometers per second
    flux : float
        Total integrated flux of the line to be added
    """
    ckms = constants.c.value / 1e3
    obs_center_wl = rest_center_wl * (1 + redshift)
    sigma_aa = fwhm_kms / ckms * obs_center_wl / (2. * np.sqrt(2. * np.log(2)))
    spec = gauss(x, sigma_aa, obs_center_wl) * flux
    return spec
