'''
Dispersion relation stuff

ADD REF HERE

'''

import numpy as np
from scipy import optimize


def fDispersionSTOKES5(h, H1, T):
    """
    Solves the progressive wave dispersion equation

    Args:
        h (np.ndarray): depth [m]
        H1 (np.ndarray):  wave height [m]
        T (np.ndarray): wave period

    Returns:
        np.array: wave number k [1/m]
    """

    g = 9.81
    omega = 2 * np.pi / T

    f = lambda k: _progressive_dispersion(k, H1, omega)

    k = optimize.bisect(f, 1e-7, 1)

    return k, omega


def _progressive_dispersion(k, H1, omega):
    g = 9.81
    return 1 + (H1 ** 2 * k ** 2) / 8+(H1 ** 4 * k ** 4) / 128 - omega / ((g * k) ** 0.5)


def alt_solve_dispersion(omega: float, d: float):
    """uses method of (Guo, 2002) to solve dispersion relation for k

    Args:
        omega (float): angular frequency [s^-1]
        d (float): water depth [m]

    Returns:
        k (float): wave number [m^-1]
    """

    g = 9.81
    beta = 2.4901

    x = d * omega / np.sqrt(g * d)

    y = x**2 * (1 - np.exp(-x**beta))**(-1/beta)

    k = y / d

    return k


def solve_dispersion(omega: float, h: float, upp: float):
    """returns wave number k for given angular frequency omega
    Args:
        omega (float): angular frequency [s^-1]
        h (float): water depth [metres]
        upp (float): upper limit of interval to find k over []

    Returns:
        k (float): wave number [m^-1]
    """

    k = optimize.bisect(f=_dispersion_diff, a=1e-7, b=upp, args=(h, omega))

    return k


def _dispersion_diff(k: np.ndarray, h: np.ndarray, omega: np.ndarray):
    """function to optimise in airy_dispersion

    Args:
        k (np.ndarray): wave number
        h (np.ndarray): water depth
        omega (np.ndarray): angular frequency
    """
    g = 9.81
    return omega ** 2 - g * k * np.tanh(k * h)
