'''
Code for generating wave load using Morison Loading

ADD REF HERE

'''
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from wavesim.kinematics import WaveKin
import matplotlib.pyplot as plt


@dataclass
class Load(ABC):
    """ load class
    """
    kinematics: WaveKin

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

    diameter: float = 1.0
    rho: float = 1024.0
    c_m: float = 1.0
    c_d: float = 1.0

    def compute_load(self):
        """ compute base shear time series in MN using morison load on a cylinder
        """

        F = np.empty(np.shape(self.kinematics.u))

        for i_t, t in enumerate(self.kinematics.u):
            for i_z, _ in enumerate(t):
                F[i_t, i_z] = self.rho * self.c_m * (np.pi / 4) * (self.diameter ** 2) * self.kinematics.du[i_t, i_z]\
                      + 0.5 * self.rho * self.c_d * self.diameter * self.kinematics.u[i_t, i_z]\
                      * np.abs(self.kinematics.u[i_t, i_z])

        self.load = np.sum(F, axis=1) * self.kinematics.dz / 1e6  # 1e6 converts to MN from N

        return self
