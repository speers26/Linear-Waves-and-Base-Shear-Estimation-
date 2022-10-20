from math import e
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def d_jonswap(f, hs, tp):
    """
    returns JONSWAP density for given frequency 

    Args:
        f (_type_): frequency [s^-1]
        hs (_type_): significant wave height [m]
        tp (_type_): significant wave period [s]
    """

    omega = 2 * np.pi * f
    omega_p = 2 * np.pi / tp

    dens = 5 / 16 * hs ** 2 * omega ** -5 * e ** (-5 / 4 * (omega / omega_p) ** -4)

    return dens