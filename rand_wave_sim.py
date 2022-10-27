from math import e
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import stringari_jonswap as sj

def jonswap(f, hs, tp):
    """
    returns JONSWAP density for given frequency 

    Args:
        f (_type_): frequency [s^-1]
        hs (_type_): significant wave height [m]
        tp (_type_): significant wave period [s]
    """

    fp = 1. / tp

    ## from holth book
    gamma = 5.87 * fp ** 0.86
    sigma_a = 0.0547 * fp ** 0.32
    sigma_b = 0.0783 * fp ** 0.16

    ##alpha = 1- 0.287 * np.log(gamma)## from "DNV-RP-C205 Environmental Conditions and Environmental Loads"
    ##alpha from yamaguchi - ?? who is this (got this from code i found)
    alpha = 1. / (.06533 * gamma ** .8015 + .13467) / 16.

    dens = alpha * hs ** 2 * tp ** -4 * f ** -5 * np.exp(-1.25 * (tp*f) ** -4)

    if f <= fp:
        sigma = sigma_a
    else :
        sigma = sigma_b

    norm_dens = dens * gamma ** np.exp(-0.5 * (tp * f - 1) ** 2 / sigma**2) #/ ( hs ** 2 ) * 4

    return norm_dens

def d_jonswap(f, hs, tp):
    density = jonswap(f,hs,tp) / ( hs ** 2 ) * 4
    return density

def random_waves_surface(t): 
    eta = 0
    return eta

if __name__ == "__main__":

    hs = 35
    tp = 20
    f_p = 1/tp

    n_freq = 1000
    f_seq = np.linspace(1e-3,f_p*5,n_freq)
    
    n_time = 200
    time = np.linspace(-20, 20, 200)
