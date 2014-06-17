
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
    err : float or [float, float]
        fixed error (one value given) or
        error magnitude and error variability
        (two values given, see below)
    
    Returns
    -------
    :class:`pandas.DataFrame` object
    
    Notes
    -----
    The returned dataset corresponds to
    concentration values predicted by
    the model + random perturbations.
    
    When one value is given for `err`, the
    parturbations are all generated using a
    Gaussian of mu=0 and sigma=fixed error.
    
    When two values are given for `err`, each
    perturbation is generated using a Gaussian
    of mu=0 and sigma given by another Gaussian: 
    
    mu =  sqrt(concentration) * error magnitude
    sigma = sqrt(concentration) * error variability
    
    """

    zmin, zmax = zlimits
    model_kwargs = model_kwargs or dict()
    
    depths = np.linspace(zmin, zmax, n)
    
    profile_data = pd.DataFrame()

    profile_data['depth'] = depths
    profile_data['C'] = model(profile_data['depth'],
                              *model_args,
                              **model_kwargs)

    try:
        err_magn, err_var = err
        
        err_mu = err_magn * np.sqrt(profile_data['C'])
        err_sigma = err_var * np.sqrt(profile_data['C'])

        profile_data['std'] = np.array(
            [np.random.normal(loc=mu, scale=sigma)
             for mu, sigma in zip(err_mu, err_sigma)]
            )
    except TypeError:
        profile_data['std'] = np.ones_like(depths) * err

    error = np.array([np.random.normal(scale=std)
                      for std in profile_data['std']])
    profile_data['C'] += error
    
    return profile_data