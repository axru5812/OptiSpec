import numpy as np
import pandas as pd
from scipy import ndimage
from astropy.io import fits
from astropy import constants
import pyneb as pn


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


def normalize(spec, kernel=151):
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
    s["fl"] = normalized_flux
    return s, continuum


def get_dust_correction(fluxes, lines, law="CCM89", intrinsic=2.86):
    """ Function that calculates the dust correction from measured values of Ha
    and Hb
    """
    ha = fluxes.loc["HI_6563", "value"]
    hb = fluxes.loc["HI_4861", "value"]
    rc = pn.RedCorr(law=law)
    rc.setCorr(
        (ha / hb) / intrinsic, lines.loc["HI_6563", "wl"], lines.loc["HI_4861", "wl"],
    )
    return rc


def dustcorrect_lines(fluxes, lines, law="CCM89", intrinsic=2.86):
    """
    """
    dc_fluxes = fluxes.copy()
    rc = get_dust_correction(fluxes, lines, law=law, intrinsic=intrinsic)
    for name, data in lines.iterrows():
        corr = rc.getCorr(data["wl"])
        dc_fluxes.loc[name, "value"] = corr * dc_fluxes.loc[name, "value"]

    return dc_fluxes, float(rc.E_BV)


def calculate_equivalent_widths(fluxes, lines, wl, continuum, redshift, fwhm):
    """ Calculates equivalent widths of emission lines given fluxes,
    line parameters and continuum

    Parameters
    ----------
    fluxes : pd.DataFrame
        dataframe of measured line fluxes
    lines : pd.DataFrame
        dataframe with index that is line name and columns wl for wavelength
    wl : array
        observer frame wavelength array
    redshift : float
        Redshift of galaxy
    fwhm : float
        assumed or fitted FWHM for the lines used to define the window over
        which the continuum is estimated
    """
    ews = fluxes.copy()
    rwl = wl / (1 + redshift)
    ckms = constants.c.value / 1000
    for name, data in lines.iterrows():
        window = np.where(
            (rwl > (data.wl - ((fwhm) / ckms) * data.wl))
            & (rwl < (data.wl + ((fwhm) / ckms) * data.wl))
        )
        cont = np.mean(continuum[window])
        ews.loc[name, "value"] = fluxes.loc[name, "value"] / cont
    return ews

