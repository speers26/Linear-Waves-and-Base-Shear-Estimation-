import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
import rand_wave_spatial_sim as rws


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
    gamma = 3.3
    sigma_a = 0.07
    sigma_b = 0.09

    sigma = (f < fp) * sigma_a + (f >= fp) * sigma_b

    gamma_coeff = gamma ** np.exp(-0.5 * (((f / fp - 1)/sigma) ** 2)) 
    dens = g ** 2 * (2 * np.pi) ** -4 * f ** -5 * np.exp(-1.25 * (tp*f) ** -4) * gamma_coeff

    area = sum(dens*df)

    dens *= hs ** 2 / (16 * area)

    return dens


def random_waves_surface(f: np.ndarray, t: np.ndarray, z: np.ndarray):
    """Generates random wave kinematics and surface (with x = 0)

    Args:
        f (np.ndarray): frequency [hertz]
        t (np.ndarray): time [seconds]
        z (np.ndarray): depth [metres]

    Returns:
        eta (np.ndarray): wave surface height [metres]
    """
    # old code

    # k = np.empty(n_freq) 
    # for i_f, f in enumerate(f):
    #     k[i_f] = rws.solve_dispersion(2 * np.pi * f)

    # R = np.sqrt(A ** 2 + B ** 2).reshape((1, n_freq)) 
    # outer_tf = np.outer(t,f) 
    # AoB = A / B
    # AoB = np.nan_to_num(AoB, nan = np.inf)

    # eta = np.sum(R * np.sin(outer_tf + np.arctan(AoB)), axis = 1)

    # Amp = np.sqrt(A ** 2 + B ** 2).reshape((1, n_freq)) ## not sure about this step, shouldn't affect the rest though
    # omega = 2 * np.pi * f

    np.random.seed(1)

    A = np.random.normal(0, 1, size=(1, n_freq)) * np.sqrt(dens*df)
    B = np.random.normal(0, 1, size=(1, n_freq)) * np.sqrt(dens*df)

    outer_tf = np.outer(t, f) 

    eta = np.sum(A * np.cos(2*np.pi*outer_tf) + B * np.sin(2*np.pi*outer_tf), axis=1)

    return eta


def fft_random_waves():
    """performs fft version of random wave surface calculation 

    Returns:
        eta (np.ndarray): wave surface height [metres]
    """
    np.random.seed(1)

    A = np.random.normal(0, 1, size=(1, n_freq)) * np.sqrt(dens*df) 
    B = np.random.normal(0, 1, size=(1, n_freq)) * np.sqrt(dens*df) 

    i = complex(0, 1)
    Z = A + B * i

    eta = np.real(fft(Z, n_freq))

    return eta


def random_waves_acf(tau, f):
    """find acf function of the gaussian random wave surface

    Args:
        tau (np.ndarray): lags []
        f (np.ndarray): contributing frequencies [hertz]

    Returns:
        acf (np.ndarray): auto correlation
    """

    outer_ft = np.outer(f, tau)    # (n_freq x tau_length)

    acf_mat = np.cos(2 * np.pi * outer_ft) * dens[:, np.newaxis] * df / area    # (n_freq x tau_length)
    acf = np.sum(acf_mat, axis=0)   # sum over columns to give (1 x tau_length)

    return acf


def kth_moment(k, f_seq):
    """function to return the kth moment of the JONSWAP spectrum evaulated at given frequencies

    Args:
        k (np.ndarray): moment
        f_seq (np.ndarray): frequencies [hertz]

    Returns:
        k_integral (_type_): integral equal to the kth moment
    """

    k_integral = np.sum(dens * f_seq ** k * df)

    return k_integral


if __name__ == "__main__":

    h = 100  # water depth

    hs = 35  # sig wave height
    tp = 10  # sig wave period
    f_p = 1/tp  # peak frequency

    freq = 4.0
    dt = 1/freq
    period = 100 

    nT = np.floor(period*freq)

    f_seq = np.linspace(1e-3, nT - 1, int(nT) ) / (nT / freq)
    times = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt

    dens = djonswap(f_seq, hs, tp)
    n_freq = len(f_seq)


    df = f_seq[1] - f_seq[0]
    dens = djonswap(f_seq, hs, tp)
    area = kth_moment(0, f_seq)

    t = time.time()
    eta = random_waves_surface(f_seq, times, z)
    t2 = time.time()
    dt = t2- t
    print(dt)

    t = time.time()
    eta_fft = fftshift(fft_random_waves())
    t2 = time.time()
    dt = t2 - t
    print(dt)

    spectral_estHs = 4 * np.sqrt(area)
    surface_estHs = 4 * np.std(eta)

    print(spectral_estHs, surface_estHs)
    
    tau_length = 250
    tau_seq = np.linspace(-50, 50, tau_length) ## not sure about how to pick tau range

    acf = random_waves_acf(tau_seq, f_seq)

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(f_seq, dens)
    
    plt.subplot(4, 1, 2)
    plt.plot(times, eta)

    plt.subplot(4, 1, 3)
    plt.plot(tau_seq, acf)

    plt.subplot(4, 1, 4)
    plt.plot(times, eta_fft[0])
    plt.show()