import pandas as pd
import numpy as np
from optispec import spectrum as sp
from optispec import utils
import pkg_resources
from astropy import constants
import lmfit

class Fitter:
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

    def __init__(
        self, redshift_guess, fit_redshift=True, mc_iterations=None, fwhm_guess=200, fit_config=None
    ):
        self.redshift_guess = redshift_guess
        self.fit_redshift = fit_redshift
        self.mc_iterations = mc_iterations
        self.fit_config = fit_config
        self.lines = self._read_line_list()
        self.fwhm_guess = fwhm_guess

    def fit(self, spec):
        """

        Returns
        -------
        self : returns an instance of self.
        """
        self.spec = spec
        norm, cont = sp.normalize(spec)
        self.norm = norm
        self.continuum = cont

        initial_ha = self._gen_initial_guess(spec)

        lmpars = lmfit.Parameters()
        # add redshift; tolerate 3% change
        lmpars.add(
            "redshift",
            vary=True,
            value=self.redshift_guess,
            min=self.redshift_guess * 0.97,
            max=self.redshift_guess * 1.03,
        )
        # add FWHM; toleratre min=spec.res; max = 500 km/s
        lmpars.add("fwhm", vary=True, value=self.fwhm_guess, min=50)

        # Add the lines
        for name, row in self.lines.iterrows():
            lmpars.add(
                name,
                vary=True,
                value=initial_ha * row.strength,
                min=0,
                max=initial_ha * 30,
            )

        mask = self._gen_linemask(spec.wl, self.redshift_guess)


        # Do the fitting
        minimizer = lmfit.Minimizer(
            self._residual, lmpars, fcn_args=(spec, mask)
        )
        out = minimizer.minimize(method='powell')
        self.lmfit_output = out
        self.results = self._params_to_df(out.params)
        return self

    def predict(self, x, include_continuum=True):
        """
        Returns
        -------
        spectrum : pd.DataFrame
        """
        if self.lmfit_output is not None:
            fl =  self._gen_linespec(x, self.lmfit_output.params)
            if include_continuum:
                return fl + self.continuum
            else:
                return fl
        else:
            raise ValueError('Fit not run')


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

    def _read_line_list(self):
        """Return a dataframe about the 68 different Roman Emperors.

        Contains the following fields:
            line            str, name of the spectral line
            wl              float, restframe vacuum wavelength of the line
        """
        stream = pkg_resources.resource_stream(__name__, "data/sdsslines.list")
        df = pd.read_csv(stream, delim_whitespace=True, comment="#")
        df.set_index("line", inplace=True)
        return df

    def _gen_linemask(self, wl, redshift_guess, fwhm=1500):
        """ Takes the list of lines and generates a mask array of ones and zeros
        Parameters
        ----------
        linelist : pd.DataFrame
            The lines to fit
        wl : array
            The wavelength array of the spectrum
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
        mask = np.zeros_like(wl)
        ckms = constants.c.to("km/s").value
        for line in self.lines.index:
            line_center = self.lines.loc[line].wl * (1 + redshift_guess)
            low_edge = line_center - fwhm / ckms * line_center
            high_edge = line_center + fwhm / ckms * line_center
            window_indices = (low_edge < wl) & (wl < high_edge)
            mask[window_indices] = 1
        return mask

    def _gen_linespec(self, x, pars):
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
        for i, line in self.lines.iterrows():
            linespec += utils.add_line(
                x, line.wl, reds, fwhm, pars[i]
            )

        return linespec

    def _residual(self, pars, spec, mask):
        """ Calculates the residual between the actual flux and the model
        """
        model = self._gen_linespec(spec.wl, pars)
        regions_of_interest = np.where(mask == 1)

        return spec.fl.values[regions_of_interest] - model.values[regions_of_interest]

    def _gen_initial_guess(self, spec):
        de_redshift = sp.spec_to_restframe(spec, self.redshift_guess)
        # Select Ha line and make a simple integration measurement of it
        window_size = 40  ##Å
        ref_line = "HI_6563"
        index = (spec.wl > (self.lines.loc[ref_line].wl - window_size / 2)) & (
            spec.wl < (self.lines.loc[ref_line].wl + window_size / 2)
        )
        ha_flux = np.trapz(spec.fl.values[index], spec.wl.values[index])
        return ha_flux

    def _params_to_df(self, params):
        res = []        
        for par in params:
            res.append([par, params[par].value, params[par].stderr])
        
        df = pd.DataFrame(res, columns=['name', 'value', 'err'])
        df.set_index('name', inplace=True)
        return df