from optispec import spectrum
import numpy as np
import pandas as pd
from astropy import constants

def test_read_fits():
    """ Test the fits reading function
    """
    pass


def test_read_ascii():
    """
    """
    assert True


def test_read_spectum():
    """
    """
    assert True


def test_spec_to_restframe():
    """
    """
    x = np.arange(1100, 1300)
    redshift = 1
    xshift = x * (1+redshift)
    spec = pd.DataFrame()

    spec['wl'] = xshift
    spec['fl'] = np.ones_like(x)
    spec['er'] = np.ones_like(x)

    s = spectrum.spec_to_restframe(spec, redshift)
    np.testing.assert_almost_equal(x, s.wl.values)


def test_convert_to_velocity():
    """
    """
    x = np.arange(1100, 1300)
    refwl = 1216
    redshift = 0
    v = (x / (1 + redshift) - refwl) / refwl * constants.c.to("km/s")
    spec = pd.DataFrame()

    spec['wl'] = x
    spec['fl'] = np.ones_like(x)
    spec['er'] = np.ones_like(x)

    v_spec = spectrum.convert_to_velocity(spec, refwl)

    np.testing.assert_almost_equal(v_spec.v.values, v.value)

def test_shuffle_spectrum():
    """
    """
    x = np.arange(1100, 1300)

    spec = pd.DataFrame()
    spec['wl'] = x
    spec['fl'] = 10 * np.ones_like(x)
    spec['er'] = np.ones_like(x)

    shuffled = spectrum.shuffle_spectrum(spec)

    assert np.all(shuffled.wl == spec.wl) 
    assert np.all(shuffled.er == spec.er)
    assert np.all(shuffled.fl != spec.fl)

def test_shuffle_spectrum_std():
    x = np.arange(1100, 1300)

    spec1 = pd.DataFrame()
    spec1['wl'] = x
    spec1['fl'] = 10 * np.ones_like(x)
    spec1['er'] = np.ones_like(x)

    spec2 = pd.DataFrame()
    spec2['wl'] = x
    spec2['fl'] = 10 * np.ones_like(x)
    spec2['er'] = 10 * np.ones_like(x)

    shuff1 = spectrum.shuffle_spectrum(spec1)
    shuff2 = spectrum.shuffle_spectrum(spec2)

    assert np.std(shuff1.fl) < np.std(shuff2.fl)

def test_shuffle_spectrum_distribution():
    x = np.array([1000])

    spec1 = pd.DataFrame()
    spec1['wl'] = x
    spec1['fl'] = 10 * np.ones_like(x)
    spec1['er'] = np.ones_like(x)

    # Lets get a sample
    fluxes = []
    for i in range(10000):
        s = spectrum.shuffle_spectrum(spec1)
        fluxes.append(s.fl.values)
    
    np.testing.assert_almost_equal(np.mean(fluxes), 10, decimal=2)
    np.testing.assert_almost_equal(np.std(fluxes), 1, decimal=2)

def test_normalize():
    x = np.arange(1100, 1300)

    spec1 = pd.DataFrame()
    spec1['wl'] = x
    spec1['fl'] = 10 * np.ones_like(x)
    spec1['er'] = np.ones_like(x)

    norm, cont = spectrum.normalize(spec1)

    np.testing.assert_allclose(norm, np.zeros_like(x))
    np.testing.assert_allclose(cont, spec1.fl)