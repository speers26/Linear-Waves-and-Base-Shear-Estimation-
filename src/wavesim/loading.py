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
        c_m (np.ndarray): coefficient of mass
        c_d (np.ndarray): coefficient of drag
    """

    c_d: np.ndarray
    c_m: np.ndarray
    diameter: float = 1.0
    rho: float = 1024.0

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

@dataclass
class MorisonLoadDuo(AbstractLoad):
    """ Morison load class, for two cylinders at the same location

    Args:
        diameter (float): diameter of cylinder to calculate morison loading on
        rho (float): density of fluid
        c_m_1 (np.ndarray): coefficient of mass for first cylinder
        c_d_1 (np.ndarray): coefficient of drag for first cylinder
        c_m_2 (np.ndarray): coefficient of mass for second cylinder
        c_d_2 (np.ndarray): coefficient of drag for second cylinder
    """

    cd1: np.ndarray
    cm1: np.ndarray
    cd2: np.ndarray
    cm2: np.ndarray
    diameter: float = 1.0
    rho: float = 1024.0

    def compute_load(self) -> MorisonLoadDuo:
        """compute base shear time series in MN using morison load on a cylinder

        Returns:
            MorisonLoad: returns self
        """

        F1 = self.rho * np.expand_dims(self.cm1, axis=(0, 2)) * (np.pi / 4) * (self.diameter ** 2) * self.kinematics.du\
            + 0.5 * self.rho * np.expand_dims(self.cd1, axis=(0, 2)) * self.diameter * self.kinematics.u\
            * np.abs(self.kinematics.u)

        F2 = self.rho * np.expand_dims(self.cm2, axis=(0, 2)) * (np.pi / 4) * (self.diameter ** 2) * self.kinematics.du\
            + 0.5 * self.rho * np.expand_dims(self.cd2, axis=(0, 2)) * self.diameter * self.kinematics.u\
            * np.abs(self.kinematics.u)

        self.load = np.sum(F1 + F2, axis=1) * self.kinematics.dz / 1e6