'''
Code for generating pdf, cdf of Rayleigh distribution

'''
import numpy as np


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


def rayleigh_pdf(eta: np.ndarray, hs: float):
    """_summary_

    Args:
        eta (np.ndarray): _description_
        hs (float): _description_
    """

    d = -np.exp(-8 * eta**2 / hs**2) * -8 * 2 * eta / hs**2

    return d
