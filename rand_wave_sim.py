from math import e
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import stringari_jonswap as sj

def d_jonswap(f, hs, tp):
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

    #alpha = 1- 0.287 * np.log(gamma)
    ## alpha from yamaguchi - ?? who is this
    alpha = 1. / (.06533 * gamma ** .8015 + .13467) / 16.

    dens = alpha * hs ** 2 * tp ** -4 * f ** -5 * np.exp(-1.25 * (tp*f) ** -4)

    if f <= fp:
        sigma = sigma_a
    else :
        sigma = sigma_b

    norm_dens = dens * gamma ** np.exp(-0.5 * (tp * f - 1) ** 2 / sigma**2)

    return norm_dens

if __name__ == "__main__":

    hs = 10
    tp = 5.0

    n_freq = 50
    f_seq = np.linspace(1e-3,1,n_freq)

    f_dens = np.empty(n_freq)

    for i_f, f in enumerate(f_seq):
        f_dens[i_f] = d_jonswap(f, hs ,tp)

    plt.plot(f_seq,f_dens)
    plt.axvline(x=1/tp)
    plt.xlabel('Frequency')
    plt.ylabel('Density')
    plt.show()
    