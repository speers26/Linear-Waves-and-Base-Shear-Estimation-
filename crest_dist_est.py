import numpy as np
import matplotlib.pyplot as plt
import fft_rand_wave_sim as rwave  # for random wave sim
import rand_wave_spatial_sim as rws  # for dispersion relation


def djonswap(f, hs, tp):

    """
    returns JONSWAP density for given frequency

    Args:
        f (np.ndarray): frequency [s^-1]
        hs (np.ndarray): significant wave height [m]
        tp (np.ndarray): significant wave period [s]

    Returns:
        dens (np.ndarray): JONSWAP density for given frequencies []
    """
    g = 9.81
    fp = 1. / tp
    df = f[1] - f[0]

    # example constants from https://wikiwaves.org/Ocean-Wave_Spectra
    gamma = 2  # 3.3
    sigma_a = 0.07
    sigma_b = 0.09

    sigma = (f < fp) * sigma_a + (f >= fp) * sigma_b

    gamma_coeff = gamma ** np.exp(-0.5 * (((f / fp - 1)/sigma) ** 2))
    dens = g ** 2 * (2 * np.pi) ** -4 * f ** -5 * np.exp(-1.25 * (tp*f) ** -4) * gamma_coeff

    area = sum(dens*df)

    dens *= hs ** 2 / (16 * area)

    return dens


def rayleigh_cdf(eta: np.ndarray, hs: float):
    """returns the rayleigh cdf

    Args:
        eta (np.ndarray): crest heights
        hs (float): sig wave height

    Returns:
        p (np.ndarray): rayleigh probability
    """

    p = 1 - np.exp(-8 * eta**2 / hs**2)

    return p


if __name__ == "__main__":

    hs = 10
    tp = 12

    cond = False

    CoH = np.linspace(1e-3, 1.5)

    crest_cdf = rayleigh_cdf(CoH * hs, hs)

    plt.figure()
    plt.plot(CoH, crest_cdf)
    plt.show()
