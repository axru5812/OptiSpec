import pandas as pd
import numpy as np
import .spectrum as sp
from . import utils

class Fitter():
    """
    Parameters
    ----------
    guess_redshift : float

    fit_redshift : bool

    mc_iterations : int or None

    Attributes
    ----------

    Methods
    -------
    """

    def __init__(self, guess_redshift, fit_redshift=True, mc_iterations=None, fit_config=None):
        self.guess_redshift = guess_redshift
        self.fit_redshift = fit_redshift
        self.mc_iterations = mc_iterations
        self.fit_config = fit_config

    def fit(self, x, y, errors):
        """


        Returns
        -------
        self : returns an instance of self.
        """
        pass

    def predict(self, x):
        """
        Returns
        -------
        spectrum : pd.DataFrame
        """
        pass

    def evaluate(self):
        """ Returns a series of fit evaluation metrics
        """ 
        pass

    def _setup_fit(self,):
        """
        """
        pass

    def _validate_config(self, config):
        """ Check that the config dict contains values we agree with

        Return
        ------
        config : dict
            validated config
        
        Raises
        ------
        Valueerror
        """
        pass

    def _read_line_list(self, filename):
        """
        """
        pass

    def _gen_linespec(self, pars, neblines):
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
        reds = pars["reds"]
        fwhm = pars["fwhm"]

        linespec = np.zeros_like(self.x)
        for line in neblines.index.values:
            linespec += utils.add_line(
                x, neblines.loc[line]["wavelength"], reds, fwhm, pars[line]
            )

        return linespec