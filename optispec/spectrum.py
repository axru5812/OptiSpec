import numpy as np
import pandas as pd
from scipy import ndimage
from astropy.io import fits
from astropy import constants


def read_spectrum(filename):
    """ Reads spectrum file into pandas dataframe
    """

    if ".fits" in filename:
        spec = read_fits(filename)
    else:
        spec = read_ascii(filename)

    return spec


def read_fits(filename):
    """ Reads spectrum from fits file. Current function assumes format according to SDSS spectrum.
    """
    full_spec = fits.getdata(filename)
    spec = pd.DataFrame()
    spec["wl"] = 10 ** full_spec["loglam"]
    spec["fl"] = full_spec["flux"]
    spec["er"] = 1 / np.sqrt(full_spec["ivar"])
    return spec


def read_ascii(filename, ivar=False):
    """ Read whitespace
    """
    spec = pd.read_csv(
        filename, delim_whitespace=True, comment="#", names=["wl", "fl", "er"]
    )
    if ivar:
        spec.er = 1 / np.sqrt(spec.er)
    return spec


def spec_to_restframe(spec, redshift):
    """ Function to put spectrum into restframe
    """
    s = spec.copy()
    s.wl = s.wl / (1 + redshift)
    return s


def convert_to_velocity(spec, reference_wl):
    """
    """
    v = (spec.wl / reference_wl - 1.0) * constants.c.value / 1000.0

    s = pd.DataFrame()

    s["v"] = v
    s["fl"] = spec.fl
    s["er"] = spec.er

    return s


def shuffle_spectrum(spec):
    """ Randomize the spectrum according to the errors
    """
    s = spec.copy()
    s["fl"] = np.random.normal(loc=spec.fl, scale=spec.er)

    return s


def normalize(spec, kernel=201):
    """
    Function that normalizes the input spectrum using median filtering

    Parameters
    ----------
    spec : DataFrame
        needs to contain 'fl' column

    Returns
    -------
    normalized_flux : ndarray
        Normalized flux array
    
    continuum : array
        subtracted continuum
    """
    continuum = ndimage.median_filter(spec.fl, size=kernel)
    normalized_flux = spec.fl - continuum
    s = spec.copy()
    s['fl'] = normalized_flux
    return s, continuum