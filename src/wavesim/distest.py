'''
Code here for condtioned on sea-state distribution estimation classes

i.e. estimation of cdfs for C|θ, R|θ, etc

'''

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from wavesim.kinematics import LinearKin
from wavesim.loading import AbstractLoad, MorisonLoad
from wavesim.spectrum import SeaState
from wavesim.crestdistributions import rayleigh_pdf
from dataclasses import dataclass
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from numpy import diff


@dataclass
class AbstractDistEst(ABC):
    """superclass for importance sampled distribution estimation

    Args:
        sea_state (SeaState): sea states to use for kinematic calculation
        loadType (AbstractLoad): type of loading to use for distribution estimation
        z_values (np.ndarray): z_values to calculate kinematics at
        sim_frequency (float): frequency of linear wave simulation
        sim_min (float): length of conditioned simulations
    """

    sea_state: SeaState
    z_values: np.ndarray
    sim_frequency: float = 4.0
    sim_min: float = 2.0

    @property
    def dz(self) -> float:
        """returns the step in depth points (homogenous)

        Returns:
            float: step in depth evaluation points
        """
        return self.z_values[1] - self.z_values[0]

    @property
    def sim_period(self) -> float:
        """returns the total length of time per sim

        Returns:
            float: sim period [s]
        """
        return 60*self.sim_min

    @property
    def sim_per_state(self) -> float:
        """returns the number of simulations per full length sea state

        Returns:
            float: simulations per full sea state
        """
        return self.sea_state.hours * 60 / self.sim_min

    @property
    def waves_per_sim(self) -> float:
        """returns the number of waves per conditioned simulation

        Returns:
            float: waves per conditioned simulation
        """
        return self.sim_period / self.sea_state.tp[0]

    def compute_cond_crests(self, up_CoH: np.ndarray = 2) -> None:
        """ get crests to condition on
        """
        self.CoH = np.sort(np.random.uniform(low=0, high=up_CoH, size=self.sea_state.num_SS))
        self.cond_crests = self.sea_state.hs[0] * self.CoH
        self.g = 1/(up_CoH*self.sea_state.hs[0])
        return None

    def compute_kinematics(self) -> None:
        """ get kinematics
        """
        self.kinematics = LinearKin(self.sim_frequency, self.sim_period, self.z_values, self.sea_state)
        self.kinematics.compute_spectrum()
        self.kinematics.compute_kinematics(cond=True, a=self.cond_crests)
        return None

    @abstractmethod
    def compute_sea_state_max(self) -> AbstractDistEst:
        """gets the relevant sea-state maxes
        """

    def compute_is_distribution(self, X: np.ndarray = None) -> None:
        """computes importance sampled distribution

        Args:
            X (np.ndarray): values to compute distribution at
        """

        if X is None:
            X = np.linspace(min(self.max_series), max(self.max_series), num=100)
        self.X = X

        f = rayleigh_pdf(self.cond_crests, self.sea_state.hs)
        fog = f/self.g

        cdf_unnorm = np.sum((X[:, None] > self.max_series[None, :])*fog, axis=1)/np.sum(fog)

        self.cdf = cdf_unnorm**(self.sim_per_state*self.waves_per_sim)

        return None

    def compute_density(self) -> None:
        """computes the pdf by numerically differentiating the importance sampled cdf

        pdf is normalised to integrate to 1

        Args:
            X (np.ndarray): evaluation points
        """

        self.dx = self.X[1] - self.X[0]
        self.mids = (self.X[1:] + self.X[:-1]) / 2
        unn_pdf = diff(self.cdf)/self.dx
        self.pdf = unn_pdf / (np.sum(unn_pdf * self.dx))

        return None

    def eval_pdf(self, X: np.ndarray) -> float:
        """evaluate the stored pdf estimate at specified points

        #TODO: vectorise this

        Args:
            X (np.ndarray): evaluation points

        Returns:
            float: estimated density at evaluation point
        """

        pdf = np.empty(len(X))

        for i, x in enumerate(X):
            abs_diffs = np.abs(self.mids-x)
            close_mid_ind = np.where(abs_diffs == np.min(abs_diffs))
            pdf[i] = self.pdf[close_mid_ind]

        return pdf

    def plot_distribution(self, log=True) -> None:
        """ plot the stored distribution

        Args:
            log (bool): boolean which decides if we plot cdf or log cdf
        """
        plt.figure()
        if log:
            plt.plot(self.X, np.log10(1-self.cdf))
            plt.xlabel('X')
            plt.ylabel('log10(1-p)')
        else:
            plt.plot(self.X, self.cdf)
            plt.xlabel('X')
            plt.ylabel('p')
        plt.show()

    def plot_density(self) -> None:
        """plots the stored density
        """
        plt.figure()
        plt.plot(self.mids, self.pdf, 'o')
        plt.show()


@dataclass
class CrestDistEst(AbstractDistEst):
    """sea-state max crest distribution class
    """

    def compute_sea_state_max(self) -> None:
        self.max_series = np.empty(self.sea_state.num_SS)
        for s in range(self.sea_state.num_SS):

            crests, _, _, _, _ = self.kinematics.retrieve_kinematics()
            crests = crests[:, s]
            # get maximums
            mins = argrelextrema(crests, np.less)[0]
            lower_min = np.max(mins[mins < self.kinematics.nt/2])
            upper_min = np.min(mins[mins > self.kinematics.nt/2])
            slice = crests[lower_min:upper_min]
            self.max_series[s] = max(slice)

        return None


@dataclass
class LoadDistEst(AbstractDistEst):
    """sea-state max load distribution class

    Args:
        load_type (AbstractLoad): type of loading to use
    """

    load_type: AbstractLoad = MorisonLoad

    def compute_load(self) -> None:
        """compute loading from kinematics
        """
        self.load = MorisonLoad(self.kinematics)
        self.load.compute_load()

        return None

    def compute_sea_state_max(self) -> None:
        self.max_series = np.empty(self.sea_state.num_SS)
        for s in range(self.sea_state.num_SS):

            load = self.load.retrieve_load()
            load = load[:, s]
            # get maximums
            mins = argrelextrema(load, np.less)[0]
            lower_min = np.max(mins[mins < self.kinematics.nt/2])
            upper_min = np.min(mins[mins > self.kinematics.nt/2])
            slice = load[lower_min:upper_min]
            self.max_series[s] = max(slice)

        return None
