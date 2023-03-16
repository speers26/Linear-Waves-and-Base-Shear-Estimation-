'''
Code for generating wave load using Morison Loading

ADD REF HERE

'''
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from wavesim.kinematics import WaveKin
import matplotlib.pyplot as plt


def morison_load(u, du, diameter=1.0, rho=1024.0, c_m=1.0, c_d=1.0):
    """compute unit Morison load for a vertical cylinder

    Args:
        u (np.ndarray): horizontal velocity [m/s]
        du (np.ndarray): horizontal acceleration [m/s^2]
        diameter (float, optional): _description_. Defaults to 1.0. [m]
        rho (float, optional): _description_. Defaults to 1024.0. [kg/m^3]
        c_m (float, optional): _description_. Defaults to 1.0. [unitless]
        c_d (float, optional): _description_. Defaults to 1.0. [unitless]

    Returns:
        np.ndarray: horizontal unit morrison load [N/m]
    """

    return rho * c_m * (np.pi / 4) * (diameter ** 2) * du + 0.5 * rho * c_d * diameter * u * np.abs(u)


def morison_base_shear(u, du, dz):
    """ compute base shear time series in MN using morison load on a cylinder

    Args:
        t_values (np.ndarray): time values to return series at [s]
        z_values (np.ndarray): z_values to integrate morison loading over [m]
        u (np.ndarray): horizontal velocities [ms^-1]
        du (np.ndarray): horizontal accelerations [ms^-2]

    Returns:
        base_shear (np.ndarray): time series of base shear forces [MN]
    """

    F = np.empty(np.shape(u))

    for i_t, t in enumerate(u):
        for i_z, z in enumerate(t):
            F[i_t, i_z] = morison_load(u[i_t, i_z], du[i_t, i_z])

    base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N

    return base_shear


@dataclass
class Load(ABC):
    """ load class
    """
    kinematics: WaveKin
    load: np.ndarray = 0

    @abstractmethod
    def compute_load(self):
        """ compute load at individual z points in WaveKin

        store in load
        """

    def retrieve_load(self):
        """ retrieve the forces stores in load"""
        return self.load

    def plot_load(self):
        """ plot the force stored in load """
        plt.plot()
        plt.plot(self.kinematics.t_values, self.load)
        plt.show()


@dataclass
class MorisonLoad(Load):
    """ Morison load class """

    def compute_load(self):
        self.load = morison_base_shear(self.kinematics.u, self.kinematics.du, self.kinematics.dz)
        return self