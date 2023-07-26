'''
Code for generating wave load using Morison Loading

ADD REF HERE

'''
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from wavesim.kinematics import AbstractWaveKin, LinearKin
import matplotlib.pyplot as plt
from wavesim.spectrum import Jonswap
from scipy.signal import argrelextrema
import wavesim.crestdistributions as crestd


@dataclass
class AbstractLoad(ABC):
    """ load class

    Args:
        kinematics (WaveKin): class of wave kinematics to use when computing load
    """
    kinematics: AbstractWaveKin

    @abstractmethod
    def compute_load(self) -> AbstractLoad:
        """compute load at individual z points in WaveKin

        store in load


        Returns:
            AbstractLoad: returns self
        """

    def retrieve_load(self) -> np.ndarray:
        """retrieve the forces stores in load

        Returns:
            np.ndarray: induced load time series
        """
        return self.load

    def plot_load(self, s=0) -> None:
        """ plot the force stored in load

        Args:
            s (int): index of sea-state to plot load for
        """
        plt.plot()
        plt.plot(self.kinematics.t_values, self.load[:, s])
        plt.ylabel("Force [MN]")
        plt.xlabel("Time")
        plt.show()


@dataclass
class MorisonLoad(AbstractLoad):
    """ Morison load class

    Args:
        diameter (float): diameter of cylinder to calculate morison loading on
        rho (float): density of fluid
        c_m (float): coefficient of mass
        c_d (float): coefficient of drag
    """

    diameter: float = 1.0
    rho: float = 1024.0
    cm_l: float = 1.0
    cm_u: float = 100.0
    cd_l: float = 1.0
    cd_u: float = 100.0
    deck_height: float = 25.0

    @property
    def c_m(self) -> np.ndarray:
        """creates vector of varying c_m with a single changepoint (representing wave-in-deck loading)

        Returns:
            np.ndarray: c_m constant vector
        """

        diffs = abs(self.kinematics.z_values-self.deck_height)

        deck_ind = np.where(diffs == np.min(diffs))[0][0]

        c_m = np.concatenate((np.tile(self.cm_l, deck_ind), np.tile(self.cm_u, self.kinematics.nz-deck_ind)))

        return c_m

    @property
    def c_d(self) -> np.ndarray:
        """creates vector of varying c_m with a single changepoint (representing wave-in-deck loading)

        Returns:
            np.ndarray: c_d constant vector
        """

        diffs = np.abs(self.kinematics.z_values-self.deck_height)

        deck_ind = np.where(diffs == np.min(diffs))[0][0]

        c_d = np.concatenate((np.tile(self.cm_l, deck_ind), np.tile(self.cm_u, self.kinematics.nz-deck_ind)))

        return c_d

    def compute_load(self) -> MorisonLoad:
        """compute base shear time series in MN using morison load on a cylinder

        Returns:
            MorisonLoad: returns self
        """

        F = self.rho * np.expand_dims(self.c_m, axis=(0, 2)) * (np.pi / 4) * (self.diameter ** 2) * self.kinematics.du\
            + 0.5 * self.rho * np.expand_dims(self.c_d, axis=(0, 2)) * self.diameter * self.kinematics.u\
            * np.abs(self.kinematics.u)

        self.load = np.sum(F, axis=1) * self.kinematics.dz / 1e6  # 1e6 converts to MN from N

        return self
