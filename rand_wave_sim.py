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

def random_waves_kinematics(h, x, t, z): ##TODO finish this
    n_freq = 1000
    f_seq = np.linspace(1e-3,f_p*5,n_freq)

    f_dens = np.empty(n_freq)

    g = 9.81
    
    for i_f, f in enumerate(f_seq):
        f_dens[i_f] = jonswap(f, hs ,tp)

   
    k, omega = f_seq * 2 * np.pi

    eta = np.sum(f_dens * np.sin(omega * t - k * x))

def airy_dispersion(h:np.ndarray,T:np.ndarray):
    """solves dispersion relation for wave number

    Args:
        h (np.ndarray): water depth
        T (np.ndarray): period [s]
    """

    omega = 2* np.pi / T

    f = lambda k: dispersion_diff(k,h,omega)

    k = optimize.bisect(f, 1e-7, 1)

    return k, omega


if __name__ == "__main__":

    hs = 35
    tp = 20
    f_p = 1/tp

    n_freq = 1000
    f_seq = np.linspace(1e-3,f_p*5,n_freq)

    f_dens = np.empty(n_freq)

    for i_f, f in enumerate(f_seq):
        f_dens[i_f] = jonswap(f, hs ,tp)

    plt.plot(f_seq,f_dens)
    plt.axvline(x=1/tp)
    plt.xlabel('Frequency')
    plt.ylabel('Density')
    plt.show()

    x = 0
    h = 100

    n_depth = 151
    z_range = np.linspace(-h, 50, n_depth)
    dz = z_range[1] - z_range[0] 

    n_time = 200
    time = np.linspace(-20, 20, 200)
