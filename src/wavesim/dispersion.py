'''
Code for solving dispersion relation to give wave number

ADD REF HERE

'''

from __future__ import annotations
import numpy as np
from scipy import optimize


def fDispersionSTOKES5(h, H, omega) -> float:
    """
    Solves the progressive wave dispersion equation

    Args:
        h (np.ndarray): depth [m]
        H (np.ndarray):  wave height [m]
        T (np.ndarray): wave period

    Returns:
        float: wave number k [1/m]
    """
    # TODO: why isnt this using depth

    f = lambda k: _progressive_dispersion(k, H, omega)

    k = optimize.bisect(f, 1e-7, 1)

    return k


def _progressive_dispersion(k, H, omega) -> float:
    g = 9.81
    return 1 + (H ** 2 * k ** 2) / 8+(H ** 4 * k ** 4) / 128 - omega / ((g * k) ** 0.5)


def alt_solve_dispersion(omega: np.ndarray, d: float) -> float:
    """uses method of (Guo, 2002) to solve dispersion relation for k

    Args:
        omega (np.ndarray): angular frequency [s^-1]
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


def solve_dispersion(omega: float, h: float, upp: float) -> float:
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


def _dispersion_diff(k: np.ndarray, h: np.ndarray, omega: np.ndarray) -> float:
    """function to optimise in airy_dispersion

    Args:
        k (np.ndarray): wave number
        h (np.ndarray): water depth
        omega (np.ndarray): angular frequency

    Returns:
        float: dispersion difference
    """
    g = 9.81
    return omega ** 2 - g * k * np.tanh(k * h)
