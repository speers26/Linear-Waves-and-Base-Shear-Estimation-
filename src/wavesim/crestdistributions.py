'''
Code for generating pdf, cdf of Rayleigh distribution

'''
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


def rayleigh_cdf(eta: np.ndarray, hs: float) -> np.ndarray:
    """returns the rayleigh cdf

    Args:
        eta (np.ndarray): evaluation points (crest heights)
        hs (float): sig wave height

    Returns:
        p (np.ndarray): rayleigh probability
    """

    p = 1 - np.exp(-8 * eta**2 / hs**2)

    return p


def rayleigh_pdf(eta: np.ndarray, hs: float) -> np.ndarray:
    """returns the rayleigh pdf

    Args:
        eta (np.ndarray): evaluation points (crest heights)
        hs (float): significant wave height

    Returns:
        p (np.ndarray): rayleigh density
    """

    d = -np.exp(-8 * eta**2 / hs**2) * -8 * 2 * eta / hs**2

    return d


@dataclass
class CrestDistribution(ABC):
    """ Super class over crest distributions

    Args:
        hs (np.ndarray): significant wave height [m]
    """
    hs: np.ndarray

    @abstractmethod
    def cdf(self, X) -> np.ndarray:
        """returns the crest cdf

        Args:
            X (np.ndarray): location at which to compute distribution

        Returns:
            (np.ndarray): distribution probability
        """

    @abstractmethod
    def pdf(self, X) -> np.ndarray:
        """returns the crest pdf

        Args:
            X (np.ndarray): location at which to compute distribution

        Returns:
            (np.ndarray): distribution density
        """


@dataclass
class Rayleigh(CrestDistribution):
    """ Class for the Rayleigh distribution
    """
    def cdf(self, X: np.ndarray) -> np.ndarray:
        """computes the rayleigh cdf

        Args:
            X (np.ndarray): evaluation points

        Returns:
            np.ndarray: distribution cdf
        """

        return 1 - np.exp(-8 * X ** 2 / (self.hs ** 2))

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """computes the rayleigh density

        Args:
            X (np.ndarray): evaluation points

        Returns:
            np.ndarray: distribution density
        """

        hs_sq = self.hs ** 2

        return -np.exp(-8 * X**2 / hs_sq) * -8 * 2 * X / hs_sq
