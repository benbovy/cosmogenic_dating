
"""
Models for calculating cosmogenic nucleide concentration
profiles

Author: B. Bovy
"""

import math

import numpy as np


def _particle_contrib(depth, erosion, exposure, density,
                      nucleide, particle):
    """
    Contribution of a specific particle type
    to the nucleide concentration.
    """
    return (
        particle['prod_rate'] / ((density * erosion / 
                                  particle['damping_depth']) 
                                 + nucleide['rdecay']) *
        np.exp(-1. * density * depth / particle['damping_depth']) *
        (1. - np.exp(-1. * exposure * ((erosion * density / 
                                       particle['damping_depth'])
                                       + nucleide['rdecay'])))
    )


def C_nucleide(depth, erosion, exposure, density,
               inheritance, nucleide, particles):
    """
    Calculate the concentration(s) of a nucleide
    at given depth(s) (generic function).
    
    Parameters
    ----------
    depth : float or array_like
        depth(s) below the surface [cm]
    erosion : float or array_like
        erosion rate(s) [cm yr-1]
    exposure : float or array_like
        exposure time [yr]
    density : float or array_like
        soil density [g cm-3]
    inheritance : float or array_like
        concentration of the nucleide at the
        initiation of the exposure scenario
        [atoms g-1]
    nucleide : dict
        nucleide parameters
    particles : [dict, dict, ...]
        parameters related to each particule type
        that contribute to the nucleide production
    
    Returns
    -------
    float or array-like (broadcasted)
        the nucleide concentration(s) [atoms g-1]
    
    Notes
    -----
    if arrays are given for several arguments, they must
    have the same shape or must be at least be broadcastable.
    
    `nucleide` must have the following key(s):
        'rdecay': radioactive decay [yr-1]
    
    each item in `particles` must have the following keys:
        'prod_rate': surface production rate [atoms g-1 yr-1]
        'damping_depth': effective apparent attenuation depth
                         [g cm-2]
    
    """
    return (
        np.sum((_particle_contrib(depth, erosion, 
                                  exposure, density,
                                  nucleide, p)
                for p in particles),
               axis=0) +
        inheritance * np.exp(-1. * nucleide['rdecay'] * exposure)
    )


def C_10Be(depth, erosion, exposure, density, inheritance,
           P_0=5.):
    """
    A model for 10Be nucleide concentration profiles.
    
    Notes
    -----
    The following parameters are used
    (Braucher et al. 2003)
    
    10Be radioactive decay: log 2 / 1.36e6
    
    Contribution of
    - neutrons
        - production rate: 0.9785 * P_0
        - damping depth: 160
    - slow muons
        - production rate: 0.015 * P_0
        - damping depth: 1500
    - fast muons
        - production rate: 0.0065 * P_0
        - damping depth: 5300
    
    See Also
    --------
    C_nucleide
    
    """
    # nucleide parameters
    berillium10 = {'rdecay': math.log(2.) / 1.36e6}
    
    # particles parameters
    neutrons = {'prod_rate': 0.9785 * P_0,
                'damping_depth': 160.}
    slow_muons = {'prod_rate': 0.015 * P_0,
                  'damping_depth': 1500.}
    fast_muons = {'prod_rate': 0.0065 * P_0,
                  'damping_depth': 5300.}
    
    return C_nucleide(depth, erosion, exposure,
                      density, inheritance,
                      berillium10,
                      [neutrons, slow_muons, fast_muons])