import numpy as np
import matplotlib.pyplot as plt
import fft_rand_wave_sim as rwave  # for random wave sim

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

def rayleigh_pdf(eta: np.ndarray, hs:float):
    """_summary_

    Args:
        eta (np.ndarray): _description_
        hs (float): _description_
    """

    d = -np.exp(-8 * eta**2 / hs**2) * -8 * 2 * eta / hs**2

    return d


if __name__ == "__main__":

    hs = 10
    tp = 12
    depth = 100
    cond = False
    a = 0

    z_num = 150
    z_range = np.linspace(-depth, 50, z_num)
    dz = z_range[1] - z_range[0]

    num_sea_states = 1000
    sea_state_hours = 3

    # don't quite get this bit - for FFT to work
    freq = 1.00  # 3. / (2*np.pi)
    period = 60**2 * sea_state_hours  # total time range
    nT = np.floor(period*freq)  # number of time points to evaluate
    t_num = int(nT)  # to work with rest of the code

    dt = 1/freq  # time step is determined by frequency
    t_range = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt  # centering time around 0

    f_range = np.linspace(1e-3, nT - 1, int(nT)) / (nT / freq)  # selecting frequency range from 0 to freq
    om_range = f_range * (2*np.pi)

    # plotting crest cdf
    CoH = np.linspace(1e-3, 1.5)
    crest_cdf = rayleigh_cdf(CoH * hs, hs)

    plt.figure()
    plt.plot(CoH, crest_cdf)

    jnswp_dens = djonswap(f_range, hs, tp)

    np.random.seed(1234)
    max_crests = np.ndarray(num_sea_states)
    for i in range(num_sea_states):
        eta_fft, u_x_fft, u_z_fft, du_x_fft, du_z_fft = rwave.fft_random_wave_sim(z_range, depth, a, om_range, jnswp_dens, cond)
        max_crests[i] = np.max(eta_fft[0])
        print(i)

    np.savetxt('max_crests.txt', max_crests, delimiter=',')

    # plotting crest pdf
    crest_pdf = rayleigh_pdf(CoH * hs, hs)

    norm_max_crests = max_crests/np.sum(max_crests)

    plt.hist(norm_max_crests)
    plt.plot(CoH*hs, crest_pdf)

    plt.show()
