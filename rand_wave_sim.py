from math import e
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def djonswap(f, hs, tp):
    """
    returns JONSWAP density for given frequency 

    Args:
        f (_type_): frequency [s^-1]
        hs (_type_): significant wave height [m]
        tp (_type_): significant wave period [s]
    """
    g = 9.81
    fp = 1. / tp

    ## from https://wikiwaves.org/Ocean-Wave_Spectra
    gamma = 3.3
    sigma_a = 0.07 
    sigma_b = 0.09

    if f <= fp:
        sigma = sigma_a
    else :
        sigma = sigma_b

    gamma_coeff = gamma ** np.exp(-0.5 * (((f / fp - 1)/sigma) ** 2)) 
    unn_dens = g ** 2 * (2 * np.pi) ** -4 * f ** -5 * np.exp(-1.25 * (tp*f) ** -4) * gamma_coeff
    n_dens = unn_dens

    return n_dens

def random_waves_surface(t): 
    eta = 0
    return eta

if __name__ == "__main__":

    hs = 15
    tp = 10
    f_p = 1/tp

    n_freq = 500
    f_seq = np.linspace(1e-3,f_p*5,n_freq)
    df = f_seq[1] - f_seq[0]

    dens = np.empty(n_freq)

    for i_f, f in enumerate(f_seq):
        dens[i_f] = djonswap(f,hs,tp)

    area = quad(djonswap, f_seq[0], f_seq[n_freq-1], args=(hs,tp))[0]
    #area = df * sum(dens)
    alpha = (hs ** 2. / 16.) / area
    dens = dens * alpha

    area_norm = area * alpha

    estHs = 4 * np.sqrt(area_norm)

    print(estHs)

    plt.figure()
    plt.plot(f_seq,dens)
    plt.show()