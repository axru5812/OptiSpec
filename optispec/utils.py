import numpy as np
from astropy import constants
import lmfit
from functools import partial
from multiprocessing import Pool
from optispec import spectrum as sp
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


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
    normalization = 1.0 / (sigma * np.sqrt(2 * np.pi))
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
    sigma_aa = (
        fwhm_kms / ckms * obs_center_wl / (2.0 * np.sqrt(2.0 * np.log(2)))
    )
    spec = gauss(x, sigma_aa, obs_center_wl) * flux
    return spec


def convolve_to_r(spec, R):
    """
    """
    pass


def gen_linemask(spec, lines, redshift_guess, fwhm=600):
    """ Takes the list of lines and generates a mask array of ones and zeros
    Parameters
    ----------
    spec : DataFrame
        The wavelength array of the spectrum
    linelist : pd.DataFrame
        The lines to fit
    redshift_guess : float
        Estimated redshift of the source
    fwhm : float (optional, default=1500)
        Size of the regions to consider around each line

    Returns
    -------
    mask : array
        0,1 binary array where regions to consider are marked with 1.
    """
    # Generate mask
    mask = np.zeros_like(spec.wl.values)
    ckms = constants.c.to("km/s").value
    for line in lines.index:
        line_center = lines.loc[line].wl * (1 + redshift_guess)
        low_edge = line_center - (fwhm / 2) / ckms * line_center
        high_edge = line_center + (fwhm / 2) / ckms * line_center
        window_indices = (low_edge < spec.wl.values) & (
            spec.wl.values < high_edge
        )
        mask[window_indices] = 1
    return mask


def gen_linespec(x, pars, lines):
    """ Generates a linespectrum to be used when constructing the fitting
    residual with lmfit
    Parameters
    ----------
    x : array
        Wavelength array
    pars : lmfit.parameters object
        An lmfit parameters objects with the general fit parameters and the
        fitted fluxes of each line
    neblines : pd.DataFrame
        dataframe with wavelength data about the fitted lines
    """
    reds = pars["redshift"]
    fwhm = pars["fwhm"]

    linespec = np.zeros_like(x)

    for i, line in lines.iterrows():
        try:
            linespec += add_line(x, line.wl, reds, fwhm, pars[i].value)
        except KeyError:
            # THis line is not included in this particular fit
            pass

    return linespec


def gen_initial_guess(redshift_guess, lines, spec):
    de_redshift = sp.spec_to_restframe(spec, redshift_guess)
    # Select Ha line and make a simple integration measurement of it
    window_size = 10  ##Å
    ref_line = "HI_6563"
    index = (
        de_redshift.wl.values > (lines.loc[ref_line].wl - window_size / 2)
    ) & (de_redshift.wl.values < (lines.loc[ref_line].wl + window_size / 2))
    ha_flux = np.trapz(
        de_redshift.fl.values[index], x=de_redshift.wl.values[index]
    )
    return ha_flux


def residual(pars, spec, mask, lines):
    """ Calculates the residual between the actual flux and the model
    """
    model = gen_linespec(spec.wl.values, pars, lines)
    regions = np.where(mask == 1)

    return spec.fl.values[regions] - model[regions]


def setup_parameters(redshift_guess, fwhm_guess, lines, spec, init_guess=None):
    """
    """
    if init_guess is None:
        initial_ha = gen_initial_guess(redshift_guess, lines, spec)
    else:
        initial_ha = init_guess

    lmpars = lmfit.Parameters()
    # add redshift; tolerate 3% change
    lmpars.add(
        "redshift",
        vary=True,
        value=redshift_guess,
        min=redshift_guess * 0.97,
        max=redshift_guess * 1.03,
    )
    # add FWHM; toleratre min=spec.res; max = 500 km/s
    lmpars.add("fwhm", vary=True, value=fwhm_guess, min=50, max=2000)

    # Add the lines
    for name, row in lines.iterrows():
        min_wl = row.wl * (1 + redshift_guess * 0.97)
        max_wl = row.wl * (1 + redshift_guess * 1.03)
        if (min_wl >= spec.wl.min()) & (max_wl <= spec.wl.max()):
            lmpars.add(
                name,
                vary=True,
                value=initial_ha * row.strength,
                min=0,
                max=initial_ha * 30,
            )

    return lmpars


def fit_spectrum(spec, lmpars, mask, lines, method):
    """
    """
    spec = remove_nonfinite(spec)
    minimizer = lmfit.Minimizer(residual, lmpars, fcn_args=(spec, mask, lines))
    out = minimizer.minimize(method=method)
    return out


def monte_carlo_fitting(
    spec,
    niterations,
    nworkers,
    lmpars,
    mask,
    lines,
    method,
    disable_prog_bar=False,
):
    """
    """

    spectra = [sp.shuffle_spectrum(spec) for i in range(niterations)]
    with Pool(nworkers) as pool:
        results = pool.map(
            partial(
                fit_spectrum,
                lmpars=lmpars,
                mask=mask,
                lines=lines,
                method=method,
            ),
            spectra,
        )

    return results


def remove_nonfinite(spec):
    s = spec.copy()
    s["fl"] = np.where(np.isfinite(s["fl"]), s["fl"], 0)
    s["er"] = np.where(np.isfinite(s["er"]), s["er"], 1e10)

    return s
