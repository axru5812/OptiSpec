from optispec import utils
import numpy as np
import pandas as pd
from optispec import fitter

def test_gauss_location():
    x = np.arange(100)
    sigma = 2
    mu = 50
    g = utils.gauss(x, sigma, mu)

    assert np.argmax(g) == mu

def test_gauss_integral():
    x = np.arange(100)
    sigma = 2
    mu = 50
    g = utils.gauss(x, sigma, mu)

    np.testing.assert_almost_equal(np.trapz(g), 1)

def test_gauss_amplitude():
    x = np.arange(100)
    sigma = 2
    mu = 50
    g = utils.gauss(x, sigma, mu)

    np.testing.assert_almost_equal(g.max(), (1/(sigma * np.sqrt(2 * np.pi))))


def test_gauss_width():
    x = np.arange(0, 100, step=0.001)
    sigma = 2
    mu = 50
    g = utils.gauss(x, sigma, mu)

    hm = g.max() / 2
    diff = np.abs(g - hm)

    indices = np.argsort(diff)

    fwhm = np.abs(x[indices[0]]-x[indices[1]])
    calc_sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    np.testing.assert_almost_equal(sigma, calc_sigma, decimal=3)


def test_add_line_flux():
    x = np.arange(1000, 1500, 0.05)
    rest_center_wl = 1216 
    redshift = 0
    fwhm_kms = 100
    flux = 100

    spec = utils.add_line(x, rest_center_wl, redshift, fwhm_kms, flux)

    np.testing.assert_almost_equal(np.trapz(spec, x=x), flux)

def test_add_line_location():
    x = np.arange(1000, 1500, 0.05)
    rest_center_wl = 1216 
    redshift = 0
    fwhm_kms = 100
    flux = 100

    spec = utils.add_line(x, rest_center_wl, redshift, fwhm_kms, flux)

    np.testing.assert_almost_equal(x[np.argmax(spec)], rest_center_wl)

def test_gen_initial_guess():
    spec= pd.DataFrame()
    redshift = 0.1
    wl = np.arange(6000, 7000, 0.5) 
    
    rest_center_wl = 6563 
    fwhm_kms = 100
    flux = 100

    fl = utils.add_line(wl, rest_center_wl, redshift, fwhm_kms, flux)

    spec['wl'] = wl * (1+redshift)
    spec['fl'] = fl
    spec['er'] = np.zeros_like(fl)

    fit = fitter.SpectrumFitter(redshift_guess=.1)

    init_guess = utils.gen_initial_guess(redshift, fit.lines, spec)
    np.testing.assert_almost_equal(init_guess, np.trapz(fl, x=wl))
