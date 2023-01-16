import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
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


def random_waves_surface_and_kinematics(f_range: np.ndarray, t_range: np.ndarray, z_range: np.ndarray, d: float,
                                        spctrl_dens: np.ndarray):
    """Generates random wave kinematics and surface (with x = 0)

    Args:
        f_range (np.ndarray): frequencies [hertz]
        t_range (np.ndarray): times [seconds]
        z_range (np.ndarray): depths [metres]
        d (float): water depth [metres]
        spctrl_dens (np.ndarray): spectral densities for given frequencies []

    Returns:
        eta (np.ndarray): wave surface height [metres]
    """
    np.random.seed(1)

    n_times = len(t_range)
    n_z = len(z_range)
    n_freq = len(f_range)
    df = f_range[1] - f_range[0]

    A = np.random.normal(0, 1, size=(1, n_freq)) * np.sqrt(spctrl_dens*df)
    B = np.random.normal(0, 1, size=(1, n_freq)) * np.sqrt(spctrl_dens*df)

    outer_tf = np.outer(t_range, f_range)

    eta = np.sum(A * np.cos(2*np.pi*outer_tf) + B * np.sin(2*np.pi*outer_tf), axis=1)

    om_range = 2*np.pi*f_range
    k = np.empty(n_freq)

    for i_om, om in enumerate(om_range):
        k[i_om] = rws.solve_dispersion(omega=om, h=d, upp=75)

    u_x = np.empty((n_z, n_times))
    for i_z, z in enumerate(z_range):
        u_x[i_z, :] = np.sum((A * np.cos(2*np.pi*outer_tf) + B * np.sin(2*np.pi*outer_tf))
                             * om_range * (np.cosh(k*(z+d))) / (np.sinh(k*d)), axis=1)
        for i_t, _ in enumerate(t_range):
            if eta[i_t] < z or z > 0:
                u_x[i_z, i_t] = 0

    return eta, u_x


def fft_random_waves(f: np.ndarray, spctrl_dens: np.ndarray):
    """performs fft version of random wave surface calculation

    Args:
        f (np.ndarray): frequency [hertz]
        spctrl_dens (np.ndarray): spectral densities for given frequencies []

    Returns:
        eta (np.ndarray): wave surface height [metres]
    """
    np.random.seed(1)

    n_freq = len(f)
    df = f[1] - f[0]

    A = np.random.normal(0, 1, size=(1, n_freq)) * np.sqrt(spctrl_dens*df)
    B = np.random.normal(0, 1, size=(1, n_freq)) * np.sqrt(spctrl_dens*df)

    i = complex(0, 1)
    Z = A + B * i

    eta = np.real(fft(Z, n_freq))

    return eta


def random_waves_acf(tau: np.ndarray, f: np.ndarray, spctrl_dens: np.ndarray):
    """find acf function of the gaussian random wave surface with given spectrum

    Args:
        tau (np.ndarray): lags []
        f (np.ndarray): contributing frequencies [hertz]
        spctrl_dens (np.ndarray): spectral densities for given frequencies []

    Returns:
        acf (np.ndarray): auto correlation
    """

    df = f[1] - f[0]
    spctrl_area = np.sum(spctrl_dens * df)

    outer_ft = np.outer(f, tau)    # (n_freq x tau_length)

    acf_mat = np.cos(2 * np.pi * outer_ft) * spctrl_dens[:, np.newaxis] * df / spctrl_area    # (n_freq x tau_length)
    acf_vec = np.sum(acf_mat, axis=0)   # sum over columns to give (1 x tau_length)

    return acf_vec


def kth_moment(k: int, f: np.ndarray, spctrl_dens: np.ndarray):
    """function to return the kth moment of the given spectrum evaulated at given frequencies

    Args:
        k (int): moment
        f_seq (np.ndarray): frequencies [hertz]

    Returns:
        k_integral (_type_): integral equal to the kth moment
    """

    df = f[1] - f[0]

    k_integral = np.sum(spctrl_dens * f ** k * df)

    return k_integral


if __name__ == "__main__":

    depth = 100.  # water depth
    z_num = 151
    z_range = np.linspace(-depth, 50, z_num)

    hs = 35.  # sig wave height
    tp = 10.  # sig wave period
    f_p = 1/tp  # peak frequency

    freq = 4.00  # 3. / (2*np.pi)
    period = 100  # total time range
    nT = np.floor(period*freq)  # number of time points to evaluate

    dt = 1/freq  # time step is determined by frequency
    times = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt  # centering time around 0

    f_seq = np.linspace(1e-3, nT - 1, int(nT)) / (nT / freq)  # selecting frequency range from 0 to freq

    jswp_density = djonswap(f_seq, hs, tp)  # finding JONSWAP density to use as spectrum for wave surface
    jswp_area = kth_moment(0, f_seq, jswp_density)  # find area of spectrum

    timer1 = time.time()
    eta, u_x = random_waves_surface_and_kinematics(f_seq, times, z_range=z_range, d=depth, spctrl_dens=jswp_density)
    timer2 = time.time()
    dt = timer2 - timer1
    print(dt)  # print time for wave surface to be generated

    timer1 = time.time()
    eta_fft = fftshift(fft_random_waves(f_seq, jswp_density))
    timer2 = time.time()
    dt = timer2 - timer1
    print(dt)  # print time for wave surface to be generated

    spectral_estHs = 4 * np.sqrt(jswp_area)
    surface_estHs = 4 * np.std(eta)

    print(spectral_estHs, surface_estHs)  # compare estimate of Hs to theory

    tau_length = 250
    tau_seq = np.linspace(-50, 50, tau_length)

    acf = random_waves_acf(tau_seq, f_seq, jswp_density)

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(f_seq, jswp_density)

    plt.subplot(4, 1, 2)
    plt.plot(times, eta)

    plt.subplot(4, 1, 3)
    plt.plot(tau_seq, acf)

    plt.subplot(4, 1, 4)
    plt.plot(times, eta_fft[0])
    plt.show()

    plt.figure()
    z_grid, t_grid = np.meshgrid(z_range, times)
    plt.scatter(z_grid.flatten(), t_grid.flatten(),  s=1, c=u_x.flatten())
    plt.plot(times, eta, '-k')
    plt.colorbar()
    plt.show()
