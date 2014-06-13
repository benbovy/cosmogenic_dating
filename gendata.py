
"""
Create generic datasets of nucleide concentration.

"""

import numpy as np
import pandas as pd

import models


def generate_dataset(model, model_args, model_kwargs=None,
                     zlimits=[50, 500], n=10,
                     err=[20., 5.]):
    """
    Create a generic dataset of nucleide concentration
    vs. depth (for testing).
    
    Parameters
    ----------
    model : callable
        the model to use for generating the data
    model_args : list, tuple
        arguments to pass to `model`
    model_kwargs : dict
        keyword arguments to pass to `model`
    zlimits : [float, float]
        depths min and max values
    n : int
        sample size
    err : [float, float]
        error magnitude and error variability
        (see below)
    
    Returns
    -------
    :class:`pandas.DataFrame` object
    
    Notes
    -----
    The returned dataset corresponds to
    concentration values predicted by
    the model + random perturbations.
    
    Each perturbation is generated using a Gaussian
    of mu=0 and sigma given by another Gaussian: 
    
    mu =  sqrt(concentration) * error magnitude
    std = sqrt(concentration) * error variability
    
    """

    err_magnitude, err_variability = err
    zmin, zmax = zlimits
    model_kwargs = model_kwargs or dict()
    
    depths = np.linspace(zmin, zmax, n)
    
    profile_data = pd.DataFrame()

    profile_data['depth'] = depths
    profile_data['C'] = model(profile_data['depth'],
                              *model_args,
                              **model_kwargs)

    err_mu = err_magnitude * np.sqrt(profile_data['C'])
    err_sigma = err_variability * np.sqrt(profile_data['C'])

    profile_data['std'] = np.array(
        [np.random.normal(loc=mu, scale=sigma)
         for mu, sigma in zip(err_mu, err_sigma)]
        )

    error = np.array([np.random.normal(scale=std)
                      for std in profile_data['std']])
    profile_data['C'] += error
    
    return profile_data