
"""
An implementation of MLE and the Bayesian approach using the Grid-Search method.

"""

import math
import inspect
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import stats


def chi2(mprofile, oprofile, ostd):
    """
    Compute the chi-square given a measured
    concentration profile (with known measurement
    error) and predicted profile(s).
    
    Parameters
    ----------
    mprofile : 1-d or n-d array_like
        the modelled concentration profile
    oprofile : 1-d array_like
        the measured concentration profile
    ostd : 1-d array_like
        the standard deviation values associated
        to each profile measurements
    
    """
    return np.sum(np.power(oprofile - mprofile, 2) /
                  np.power(ostd, 2),
                  axis=0)

    
def likelihood(mprofile, oprofile, ostd, log=True):
    """
    Compute the (log)likelihood function
    given a measured concentration profile
    (with known measurement error) and predicted
    profile(s).
    
    Parameters
    ----------
    mprofile : 1-d or n-d array_like
        the modelled concentration profile
    oprofile : 1-d array_like
        the measured concentration profile
    ostd : 1-d array_like
        the standard deviation values associated
        to each profile measurements
    log : bool
         if True, returns the log likelihood

    """
    std_square = np.power(ostd, 2)
    
    loglike = -0.5 * (
        np.sum(np.power(oprofile - mprofile, 2)
               / std_square
               - np.log(2 * np.pi * std_square),
               axis=0)
    )
    
    if log:
        return loglike
    else:
        return np.exp(loglike) 


def ppd(likelihood, prior, log=True):
    """
    Compute the (non-normalized) (log)posterior
    probability distribution given the (log)likelihood
    and the (log)prior probability distribution.
    
    Parameters
    ----------
    likelihood : float or array_like
        the (log)likelihood function
    prior : float or array_like
        the (log)prior probability distribution
    log : bool
         Must be True if log-likelihood and
         log-prior are given
    
    """
    if log:
        return prior + likelihood
    else:
        return prior * likelihood


def create_regular_grid(*ranges):
    """
    Returns a regular grid for uniform sampling
    in the multidimensional parameter space.
    
    Parameters
    ----------
    *ranges : range, range, ...
        parameters ranges.
        
        range can be either a `slice` object 
        or a (start, end, step) 3-tuple.
        if a complex number is be given as step,
        its real part will be then interpreted as
        the number of points to sample within the
        range.
    
    Returns
    -------
    [n-d array, n-d array, ...]
        an array of grid coordinates for each parameter.
        all arrays can be broadcasted to the regular
        grid formed by all parameters.

    See Also
    --------
    :func:`numpy.ogrid`

    """
    p_slices = [r if type(r) is slice else slice(*r)
                for r in ranges]
    
    return np.ogrid[p_slices]


def integrate_over_grid(grid_step, F, axis=None):
    """
    Integrate a function over a regular grid.
    
    Parameters
    ----------
    grid_step : 1-d array_like
        the resolution (step length) of one
        (`axis`) or each (`axis=None`)
        dimension of the regular grid
    F : array_like
        values of any function to integrate that 
        have been evaluated on the nodes of the
        regular grid
    axis : int or None
        if None, integrate over the entire grid,
        otherwise integrate only over the
        specified axis
    
    Returns
    -------
    float or n-d array
        depending on `axis`, one or several
        integrals
    
    """
    if axis is None:
        V = np.prod(grid_step)
    else:
        V = grid_step
    
    return V * np.sum(F, axis=axis)


def normalize_ppd(ppd, grid_steps):
    """
    Normalize the PPD values given on
    a regular grid, so that the integral
    over the grid equals 1.
    """
    norm = integrate_over_grid(grid_steps, ppd)
    return norm, ppd / norm 


def ppd_mean(ppd, grid, grid_steps):
    """
    Compute the mean of the PPD.
    
    Parameters
    ----------
    ppd : array_like
        the sampled (and normalized) PPD
    grid : array_like
        grid coordinates, as returned by
        :func:`create_regular_grid`
    grid_steps : 1-d array_like
        the resolution (step length) of each
        dimension of the regular grid
    
    Returns
    -------
    1-d array_like
        mean values for each parameter
    
    """
    ppd_mean = [integrate_over_grid(grid_steps, ppd * grid[dim])
                for dim in range(len(grid))]
    
    return np.array(ppd_mean)


def ppd_covmat(ppd, grid, grid_steps):
    """
    Compute the covariance matrix of PPD.
    
    """
    dimensions = range(len(grid))
    
    ppd_mean = compute_ppd_mean(ppd, grid, grid_steps)
    
    CM = [[integrate_over_grid(grid_steps, ppd *
                               grid[idim] * grid[jdim])
           - ppd_mean[idim] * ppd_mean[jdim]
           for jdim in dimensions]
          for idim in dimensions]
    
    return np.array(CM)


def ppd_corrmat(covmat):
    """
    Compute the correlation matrix given
    the covariance matrix `covmat`.
    """
    CrM = [[covmat[i][j] / np.sqrt(covmat[i][i] * covmat[j][j])
            for j in range(covmat.shape[1])]
           for i in range(covmat.shape[0])]
    
    return np.array(CrM)


def marginal_ppd(ppd, grid_steps, *dims):
    """
    Compute the (joint) marginal PPD for one or
    more parameters.
    
    Parameters
    ----------
    ppd : n-d array_like
        the sampled (and normalized) PPD
    grid_steps : 1-d array_like
        the resolution (step length) of each
        dimension of the regular grid
    *dims : int, int, ...
        parameters (grid dimensions) for which
        to compute the (joint) marginal PPD
    
    Returns
    -------
    1-d or n-d array
        values of the (joint) marginal PPD
        on the regular grid. The number of
        dimensions depends on the number
        of `*dims` arguments given.
    """
    M = ppd.copy()
    ax = 0
    
    for d in range(len(grid_steps)):
        if d in dims:
            ax += 1
            continue
        M = integrate_over_grid(grid_steps[d], M, axis=ax)
    
    return M


def profile_likelihood(likelihood, *dims):
    """
    Compute the profile (log)likelihood for
    one or more parameters
    
    Parameters
    ----------
    likelihood : n-d array_like
        (log)likelihood
    grid_steps : 1-d array_like
        the resolution (step length) of each
        dimension of the regular grid
    *dims : int, int, ...
        parameters (grid dimensions) for which
        to compute the profile (log)likelihood
    
    Returns
    -------
    1-d or n-d array
        values of the profile (log)likelihood
        on the regular grid. The number of
        dimensions depends on the number
        of `*dims` arguments given.
    """
    Lp = likelihood.copy()
    ax = 0
    
    for d in range(likelihood.ndim):
        if d in dims:
            ax += 1
            continue
        Lp = Lp.max(axis=ax)
    
    return Lp


def profile_likelihood_crit(profile_likelihood,
                            max_likelihood,
                            clevels=[0.674, 0.95, 0.997],
                            log=True):
    """
    Return the critical values of the profile
    likelihood that correspond to the given confidence
    levels (based on the likelihood ratio test).
    
    Useful for the calculation of confidence intervals.
    
    Parameters
    ----------
    profile_likelihood : n-d array_like
        the profile (log)likelihood
    max_likelihood : float
        maximized value of the (log) likelihood
    clevels : list
        confidence levels
    log : bool
        must be True if log-likelihoods are
        provided
        
    """
    df = profile_likelihood.ndim
    lambda_crit = [stats.chi2(df).ppf(cl)
                   for cl in clevels]
    ploglike_crit = (2. * max_likelihood - lambda_crit) / 2.
    
    if log:
        return ploglike_crit
    else:
        return np.exp(ploglike_crit)


class CosmogenicInferenceGC():
    """
    Infer a set of parameters from measured cosmogenic
    profile(s) using either MLE or Bayesian inference
    with the grid search sampling method.
    
    Parameters
    ----------
    description : string
        brief description 
    
    """
    def __init__(self, description=''):
        
        self.description = description
        self.oprofile = dict()
        self.parameters = OrderedDict()
        self.grid = None
        self.grid_size = None
        self.grid_steps = None
        self.mprofiles = None
        self.chisq = None
        self.likelihood = None
        self.loglike = None
        self.maxlike = None
        self.mle = None
        self.ppd = None
        self.ppd_norm = None
        self.ppd_mean = None
        self.ppd_mean_f = None
        self.ppd_max = None
        self.ppd_max_i = None
        self.ppd_max_f = None
        self.ppd_covmat = None
        self.ppd_corrmat = None
        self.M_ppds_1d = None
    
    def set_profile_measured(self, depth, C, std, nucleide,
                             **kwargs):
        """
        Set the measured nucleide concentration profile.
        
        Parameters
        ----------
        depth : 1-d array_like
            the depth values
        C : 1-d array_like
            the measured nucleide concentration
            values
        std : 1-d array_like
            the standard deviation of the measured
            concentrations
        nucleide : 1-d array_like
            allow to distinguish concatenated profiles
            of multiple nucleides
        **kwargs : name=value, name=value...
            any other information to provide about
            the profile
        
        """
        self.oprofile['depth'] = np.array(depth)
        self.oprofile['C'] = np.array(C)
        self.oprofile['std'] = np.array(std)
        self.oprofile['nucleide'] = np.array(nucleide)
        self.oprofile.update(kwargs)
        
    def set_profile_model(self, func):
        """
        Set the mathematical model for predicting
        the comsogenic concentration profiles.

        Parameters
        ----------
        func : callable
            must accept depth values as the first
            argument and parameter value(s) as the
            other arguments for each parameter to fit,
            defined in the SAME ORDER than
            :attr:`CosmogenicProfileBayesGC.parameters` !
        
        """
        self.profile_model = func

    def set_parameter(self, name, srange, prior=None,
                      **kwargs):
        """
        Set a model parameter to fit.

        Parameters
        ----------
        name : string
            name of the parameter
        srange : (start, stop, step)
            parameter search range used to compute
            the sampling regular grid. if a complex
            number is given for `step`, its real part
            will be the number of samples to generate
            instead of a step length.
        prior : callable
            the prior density probability function
            for the parameter (must accept a 1-d
            array_like as unique argument)
        **kwargs : name=value, name=value...
            any other information to provide about
            the parameter
        
        """
        p = dict()
        p['range'] = srange
        p['prior'] = prior
        p.update(kwargs)
        self.parameters[name] = p
    
    @property
    def deg_freedom(self):
        try:
            return self.oprofile['C'].size - len(self.parameters)
        except Exception:
            return None

    def _set_sampling_grid(self):
        """
        Create the sampling regular grid.
        """
        ranges = [p['range'] for p in self.parameters.values()]
        
        self.grid = create_regular_grid(*ranges)
        self.grid_sizes = [a.size for a in self.grid]
        self.grid_total_size = np.prod(self.grid_sizes)

        self.grid_steps = [1. * (stop - start) / step
                           if isinstance(step, complex)
                           else step
                           for start, stop, step in ranges]
    
    def compute_mprofiles(self):
        """
        Calculate the predicted nucleide concentration
        vs. depth profiles at every node of the
        sampling grid.
        """
        if self.grid is None:
            self._set_sampling_grid()
            
        # array broadcasting...
        depth = self.oprofile['depth'].copy()
        for dim in range(len(self.grid)):
            depth = np.expand_dims(depth, axis=-1)
        grid = [np.expand_dims(p, axis=0) for p in self.grid]
        
        self.mprofiles = self.profile_model(depth, *grid)
    
    def compute_like(self, f='loglike'):
        """
        Calculate the loglikelihood (`f`='loglike'),
        likelihood (`f`='likelihood') or chi-square
        (`f`='chisq') values at every node
        of the sampling grid.
        """
        oprofile = self.oprofile['C'].copy()
        ostd = self.oprofile['std'].copy()
        
        for dim in range(len(self.grid)):
            oprofile = np.expand_dims(oprofile, axis=-1)
            ostd = np.expand_dims(ostd, axis=-1)
        
        if f == 'loglike':
            self.likelihood = likelihood(self.mprofiles,
                                         oprofile, ostd)
        elif f == 'likelihood':
            self.likelihood = likelihood(self.mprofiles,
                                         oprofile, ostd,
                                         log=False)
        elif f == 'chisq':
            self.chisq = chi2(self.mprofile,
                              oprofile, ostd) 
    
    def compute_from_data_model(self, data_model):
        """
        Get modelled profiles, chi2_r, likelihood, prior
        and ppd from the given `data_model`.
        
        Returns a dictionary with the computed values.
        
        """
        f_names = ['mprofile', 'chisq', 'chisq_r', 'loglike',
                   'prior', 'ppd']
        
        mprofile = self.profile_model(self.oprofile['depth'],
                                      *data_model)

        chisq = chi2(mprofile, self.oprofile['C'],
                     self.oprofile['std'])
        chisq_r = chisq / self.deg_freedom
        
        loglike = likelihood(mprofile, self.oprofile['C'],
                             self.oprofile['std'])
        
        prior_funcs = [p['prior'] for p in self.parameters.values()]
        prior = np.prod([pf(dm)
                         for pf, dm in zip(prior_funcs, data_model)])

        ppd = compute_ppd(prior, math.exp(loglike))
        ppd /= self.ppd_norm
        
        results = dict(
            zip(f_names, [mprofile, chisq, chisq_r,
                          loglike, prior, ppd])
        )
        
        return results
    
    
    def compute_mle(self, log=True,
                    save_mprofiles=False,
                    save_likelihood=False):
        """
        Compute the (log)likelihood, find its
        maximum, and compute 1d and 2d profile
        (log)likelihoods.
        """
        
        # compute (log)likelihood
        self.compute_mprofiles()
        
        if log:
            f = 'loglike'
        else:
            f = 'likelihood'
        self.compute_like(f=f)
        
        # find maximum
        self.maxlike = self.likelihood.max()
        
        mle_ind = np.nonzero(self.likelihood >= self.maxlike)
        self.mle = [p.flatten()[mi]
                    for p, mi in zip(self.grid, mle_ind)]
        
        # profile likelihoods
        self.proflike1d = [profile_likelihood(self.likelihood,
                                              dim)
                           for dim in range(len(self.grid))]
        
        self.proflike2d = [[profile_likelihood(self.likelihood,
                                               idim, jdim)
                            for jdim in range(len(self.grid))]
                           for idim in range(len(self.grid))]
        
        # keep or delete intermediate results
        if not save_mprofiles:
            self.mprofiles = None
        if not save_likelihood:
            self.likelihood = None 
        
    
    def compute_bayes(self, save_mprofiles=False,
                      save_likelihood=False):
        """
        Compute the normalized PPD, its mean,
        its covariance matrix and all the 1-d and
        2-d marginal PPDs (may take a while to compute
        and may consume a lot of memory, depending
        on the size of the sampling grid!!).
        
        The specified keyword arguments can be used to save
        the intermediate results in the corresponding
        attributes
        """ 
        # compute the prior distribution 
        prior_funcs = [p['prior'] for p in self.parameters.values()]
        prior = np.prod([pf(pg) for pf, pg in zip(prior_funcs, self.grid)],
                        axis=0)
        
        # compute the likelihood function
        self.compute_mprofiles()
        self.compute_like(f='likelihood')
        
        # compute and normalize the PPD
        ppd = ppd(self.likelihood, prior)
        self.ppd_norm, self.ppd = normalize_ppd(ppd, self.grid_steps)
        
        # keep or delete intermediate results
        if not save_mprofiles:
            self.mprofiles = None
        if not save_likelihood:
            self.likelihood = None
        
        del ppd
        del grid
        
        # compute PPD mean and mode (+ functions values)
        self.ppd_mean = ppd_mean(self.ppd, self.grid,
                                 self.grid_steps)
        
        self.ppd_mean_f = self.compute_from_data_model(self.ppd_mean)
        
        self.ppd_max_i = np.nonzero(self.ppd >= self.ppd.max())
        self.ppd_max = [p.flatten()[mi]
                        for p, mi in zip(self.grid,
                                         self.ppd_max_i)]

        self.ppd_max_f = self.compute_from_data_model(self.ppd_max)
        
        # compute PPD covavriance and correlation matrices
        self.ppd_covmat = ppd_covmat(self.ppd,
                                     self.grid,
                                     self.grid_steps)
        
        self.ppd_corrmat = corrmat(self.ppd_covmat)
        
        
        # compute 1D marginal PPDs and find maximums
        self.M_ppds_1d = [marginal_ppd(self.ppd,
                                       self.grid_steps,
                                       dim)
                          for dim in range(len(self.grid))]
        
        M_ppds_1d_max_i = [M.argmax()
                           for M in self.M_ppds_1d]
        self.M_ppds_1d_max = [p.flatten()[mi]
                              for p, mi in zip(self.grid,
                                               M_ppds_1d_max_i)]
        
        self.M_ppds_1d_max_f = self.compute_from_data_model(
            self.M_ppds_1d_max
        )
        
        # compute 2D marginal PPDs
        self.M_ppds_2d = [[marginal_ppd(self.ppd,
                                        self.grid_steps,
                                        idim, jdim)
                           for jdim in range(len(self.grid))]
                          for idim in range(len(self.grid))]
    
    
    def setup_summary(self):
        if self.grid is None:
            self._set_sampling_grid()
        
        summary = "Modelling C profile (Bayes, Grid-Search)\n\n"
        summary += "DESCRIPTION:\n{desc}\n\n".format(
                       desc=self.description
                   )
        summary += "MEASURED PROFILE ({N} samples):\n".format(
                       N=len(self.oprofile['C'])
                   )
        summary += str(pd.DataFrame(self.oprofile))
        summary += "\n\n"
        summary += "PROFILE MODEL:\n{fname}\n{fdoc}\n\n".format(
                       fname=self.profile_model.__name__,
                       fdoc=inspect.getdoc(self.profile_model)
                   )
        summary += "'UNKNOWN' PARAMETERS ({n}):\n".format(
                       n=len(self.parameters)
                   )
        summary += "\n".join([
                       name + ":\n" +
                       "\n".join(["\t{0}: {1}".format(k, v)
                                  for k, v in p.items()])
                        for name, p in self.parameters.items()
                   ])
        summary += "\n\ndegrees of freedom: {dof}\n\n".format(
                       dof=self.deg_freedom
                   )
        summary += "GRID SEARCH:\n"
        summary += "nb. of nodes per parameter: {np}\n".format(
                       np=self.grid_sizes
                   )
        summary += "total nb. of nodes: {ng}\n\n".format(
                       ng=self.grid_total_size
                   )
        
        return summary
    
    def results_summary(self):
        summary = "RESULTS:\n\n"
        
        if self.ppd is None:
            return summary + "no result yet"
        
        summary += "parameter names in order:\n{0}\n\n".format(
                       self.parameters.keys()
                   )
        summary += "PPD max:\n{0}\n\n".format(self.ppd_max)
        summary += "Values at PPD max:\n"
        summary += "\n".join(["{0}:\n {1}".format(k, v)
                              for k, v in self.ppd_max_f.items()])
        summary += "\n\n"
        summary += "PPD mean:\n{0}\n\n".format(self.ppd_mean)
        summary += "Values at PPD mean:\n"
        summary += "\n".join(["{0}:\n {1}".format(k, v)
                              for k, v in self.ppd_mean_f.items()])
        summary += "\n\n"
        summary += "1D Marginal PPD maxs:\n{0}\n\n".format(
                        self.M_ppds_1d_max
                   )
        summary += "Values at 1D Marginal PPD maxs:\n"
        summary += "\n".join(["{0}:\n {1}".format(k, v)
                              for k, v in self.M_ppds_1d_max_f.items()])
        summary += "\n\n"
        summary += "PPD covmat:\n{0}\n\n".format(self.ppd_covmat)
        summary += "PPD corrmat:\n{0}\n\n".format(self.ppd_corrmat)
        
        return summary
    
    def __str__(self):
        return self.setup_summary() + self.results_summary()