import pandas as pd
import numpy as np
from optispec import spectrum as sp
from optispec import utils
import pkg_resources
from astropy import constants
import lmfit


class SpectrumFitter:
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
        self,
        redshift_guess,
        fit_redshift=True,
        mc_iterations=None,
        mc_workers=4,
        fwhm_guess=200,
        method="leastsq",
    ):
        self.redshift_guess = redshift_guess
        self.fit_redshift = fit_redshift
        self.mc_iterations = mc_iterations
        self.mc_workers = mc_workers
        self.lines = self._read_line_list()
        self.fwhm_guess = fwhm_guess
        self.method = method

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

        mask = utils.gen_linemask(spec, self.lines, self.redshift_guess)

        lmpars = utils.setup_parameters(
            self.redshift_guess, self.fwhm_guess, self.lines, self.spec
        )

        # Do the fitting
        out = utils.fit_spectrum(self.spec, lmpars, mask, self.lines, self.method)
        self.lmfit_output = out
        self.results = self._params_to_df(out.params)
        self.line_results = self._line_params_to_df(out.params)

        if self.mc_iterations is not None:
            mc_res = utils.monte_carlo_fitting(
                self.spec,
                self.mc_iterations,
                self.mc_workers,
                lmpars,
                mask,
                self.lines,
                self.method,
            )
            self.mc_posteriors, self.mc_summary = self._aggregate_mc_results(mc_res)

        return self

    def _aggregate_mc_results(self, results):
        resdict = {}
        for i, lmfitoutput in enumerate(results):
            paramset = lmfitoutput.params
            for param in paramset:
                if i == 0:
                    resdict[param] = [paramset[param].value]
                else:
                    resdict[param].append(paramset[param].value)
        summary = {}
        for key in resdict.keys():
            summary[key] = np.std(np.array(resdict[key]))
        return resdict, summary

    def predict(self, x, include_continuum=True):
        """
        Returns
        -------
        spectrum : pd.DataFrame
        """
        if self.lmfit_output is not None:
            fl = utils.gen_linespec(x, self.lmfit_output.params, self.lines)
            if include_continuum:
                return fl + self.continuum
            else:
                return fl
        else:
            raise ValueError("Fit not run")

    def evaluate(self):
        """ Returns a series of fit evaluation metrics
        """
        pass

    def _setup_fit(self,):
        """
        """
        pass

    def _read_line_list(self):
        """Return a dataframe with data about the nebular lines

        Contains the following fields:
            line : str
                name of the spectral line
            wl : float
                restframe vacuum wavelength of the line
            str : float
                initial guess strength relative to Halpha            
        """
        stream = pkg_resources.resource_stream(__name__, "data/sdsslines.list")
        df = pd.read_csv(stream, delim_whitespace=True, comment="#")
        df.set_index("line", inplace=True)
        return df

    def _line_params_to_df(self, params):
        res = []
        for line, row in self.lines.iterrows():
            try:
                res.append([line, params[line].value, params[line].stderr])
            except KeyError:
                res.append([line, np.nan, np.nan])

        df = pd.DataFrame(res, columns=["name", "value", "err"])
        df.set_index("name", inplace=True)
        return df

    def _params_to_df(self, params):
        res = []
        for line, row in self.lines.iterrows():
            try:
                res.append([line, params[line].value, params[line].stderr])
            except KeyError:
                res.append([line, np.nan, np.nan])
        res.append(["redshift", params["redshift"].value, params["redshift"].stderr])
        res.append(["fwhm", params["fwhm"].value, params["fwhm"].stderr])

        df = pd.DataFrame(res, columns=["name", "value", "err"])
        df.set_index("name", inplace=True)
        return df
