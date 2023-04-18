'''
Code for generating wave load using Morison Loading

ADD REF HERE

'''
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
        plt.ylabel("Force [MN]")
        plt.xlabel("Time")
        plt.show()


@dataclass
class MorisonLoad(AbstractLoad):
    """ Morison load class

    Args:
        diameter (float): diameter of cylinder to calculate morison loading on
        rho (float): density of fluid
        c_m: float: ??? constant
        c_d: float ??? constant
    """

    diameter: float = 1.0
    rho: float = 1024.0
    c_m: float = 1.0
    c_d: float = 1.0

    def compute_load(self):
        """ compute base shear time series in MN using morison load on a cylinder
        """

        F = self.rho * self.c_m * (np.pi / 4) * (self.diameter ** 2) * self.kinematics.du\
            + 0.5 * self.rho * self.c_d * self.diameter * self.kinematics.u\
            * np.abs(self.kinematics.u)

        self.load = np.sum(F, axis=1) * self.kinematics.dz / 1e6  # 1e6 converts to MN from N

        return self


@dataclass
class LoadDistEst():  # TODO: add option to specify structure height
    """
    general estimated load class

    Args:
        hs (np.ndarray): significant wave height [m]
        tp (np.ndarray): significant wave period [s]
        num_sea_states (int): number of sea states to find max force within
        sea_state_hours (np.ndarray): length of each sea state in hours [hours]
        z_values (np.ndarray): depths to evaluate wave kinematics at [m]

        sim_frequency (np.ndarray): frequency of simulated wave kinematic observations [hertz]
        sim_min (np.ndarray): length of time to simulate for each sea state in minutes [min]
        spctrType (type): class of wave spectrum to use
        loadType (type): class of loading to use
    """
    hs: np.ndarray
    tp: np.ndarray
    num_sea_states: int
    sea_state_hours: np.ndarray
    z_values: np.ndarray

    sim_frequency: np.ndarray = 1.00  # TODO: make code work when this is 4.00
    sim_min: np.ndarray = 2.00
    spctrType: type = Jonswap
    loadType: type = MorisonLoad

    @property
    def dz(self):
        return self.z_values[1] - self.z_values[0]

    @property
    def dt(self):
        return 1/self.sim_frequency

    @property
    def sim_period(self):
        return 60*self.sim_min

    @property
    def sim_per_state(self):
        return self.sea_state_hours * 60 / self.sim_min

    @property
    def waves_per_sim(self):
        return self.sim_period / self.tp

    def compute_tf_values(self):
        """ get frequency and times for given simulation frequency and length"""
        nT = np.floor(self.sim_period*self.sim_frequency)  # number of time points to evaluate
        self.t_values = np.linspace(-nT/2, nT/2 - 1, int(nT)) * self.dt  # centering time around 0
        self.f_values = np.linspace(1e-3, nT - 1, int(nT)) / (nT / self.sim_frequency)  # selecting frequency range
        self.df = self.f_values[1] - self.f_values[0]
        self.nt = int(nT)
        return self

    def compute_spectrum(self):
        """ get spectrum for given frequencies, store in spctr"""
        self.spctr = self.spctrType(self.f_values, self.hs, self.tp)
        self.spctr.compute_density()
        return self

    def compute_cond_crests(self, up_CoH: np.ndarray = 2):
        """ get crests to condition on """
        self.CoH = np.sort(np.random.uniform(low=0, high=up_CoH, size=self.num_sea_states))
        self.cond_crests = self.hs * self.CoH
        self.g = 1/(up_CoH*self.hs)
        return self

    def simulate_sea_states(self):
        """ simulate conditioned sea states and store elevation and load
        also get and store max elevation and load
        """
        self.crests = np.empty([self.num_sea_states, self.nt])
        self.load = np.empty([self.num_sea_states, self.nt])
        self.max_crests = np.empty(self.num_sea_states)
        self.max_load = np.empty(self.num_sea_states)
        for i in range(self.num_sea_states):
            # print(i)
            lin_kin = LinearKin(t_values=self.t_values, z_values=self.z_values, spctr=self.spctr)
            lin_kin.compute_kinematics(cond=True, a=self.cond_crests[i])
            self.crests[i, :], _, _, _, _ = lin_kin.retrieve_kinematics()

            lin_load = MorisonLoad(lin_kin)
            lin_load.compute_load()
            self.load[i, :] = lin_load.retrieve_load()

            # get maximums
            mins = argrelextrema(self.crests[i, :], np.less)[0]
            lower_min = np.max(mins[mins < self.nt/2])
            upper_min = np.min(mins[mins > self.nt/2])
            slice = self.crests[i, lower_min:upper_min]
            self.max_crests[i] = max(slice)
            # do forces here for a single wave
            slice = self.load[i, lower_min:upper_min]
            self.max_load[i] = max(slice)

        return self

    def compute_crest_dist(self, X: np.ndarray):
        """compute max crest dist by importance sampling (should be rayleigh)

        Args:
            X (np.ndarray): values to compute IS crest dist at
        """
        self.crest_X = X

        f = crestd.rayleigh_pdf(self.cond_crests, self.hs)
        fog = f/self.g

        crest_cdf_unnorm = np.sum((X[:, None] > self.max_crests[None, :])*fog, axis=1)/np.sum(fog)

        self.crest_cdf = crest_cdf_unnorm**(self.sim_per_state*self.waves_per_sim)

        return self

    def compute_load_dist(self):
        """compute max load dist by importance sampling
        """

        # self.load_X = np.linspace(min(self.max_load), max(self.max_load), num=100)
        self.load_X = np.linspace(0, 3, num=100)

        f = crestd.rayleigh_pdf(self.cond_crests, self.hs)
        fog = f/self.g

        load_cdf_unnorm = np.sum((self.load_X[:, None] > self.max_load[None, :])*(fog), axis=1)/np.sum(fog)

        self.load_cdf = load_cdf_unnorm**(self.sim_per_state*self.waves_per_sim)

    def plot_crest_dist(self, log=True):
        """ plot the stored crest distribution against theoretical """
        theor_cdf = crestd.rayleigh_cdf(self.crest_X, self.hs)**(self.sim_per_state*self.waves_per_sim)
        plt.figure()
        if log:
            plt.plot(self.crest_X, np.log10(1-self.crest_cdf))
            plt.plot(self.crest_X, np.log10(1-theor_cdf), '-r')
        else:
            plt.plot(self.crest_X, self.crest_cdf)
            plt.plot(self.crest_X, theor_cdf, '-r')
        plt.show()

    def plot_load_dist(self, log=True):
        """ plot the stored load distribution """
        plt.figure()
        if log:
            plt.plot(self.load_X, np.log10(1-self.load_cdf))
            plt.xlabel("Load [MN]")
            plt.ylabel("log10(1-P)")
        else:
            plt.plot(self.load_X, self.load_cdf)
            plt.xlabel("Load [MN]")
            plt.ylabel("P")
        plt.show()
