import pandas as pd
import numpy as np
from numpy.testing import assert_almost_equal
from optispec.fitter import Fitter
from optispec import utils


def test_read_line_list():
    fitter = Fitter(0.2)

    df = fitter._read_line_list()
    assert df.wl.values[0] == 6564.61
    assert df.index.values[0] == 'HI_6563'


def test_fit():
    x = np.arange(4900, 6800, 0.8)
    redshift = 0.1
    x *= (1+redshift)

    ha_wl = 6564.61
    OIII_5007 = 5008.240 
    fwhm_kms = 300
    flux = 100

    spec= pd.DataFrame()
    fl = utils.add_line(x, ha_wl, redshift, fwhm_kms, flux)
    fl += utils.add_line(x, OIII_5007, redshift, fwhm_kms, flux)
    fl = fl + np.ones_like(x)
    spec['fl'] = np.random.normal(loc=fl, scale = 0.01*np.ones_like(x))
    spec['wl'] = x
    spec['er'] = 0.1 * fl

    fitter = Fitter(redshift * 0.99)
    fitter.fit(spec)

    assert_almost_equal(fitter.results.loc['HI_6563'].value, flux)
