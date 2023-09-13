'''
Code here for condtioned on sea-state distribution estimation classes

i.e. estimation of cdfs for C|θ, R|θ, etc

'''

from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from wavesim.kinematics import LinearKin
from wavesim.loading import MorisonLoad
from wavesim.spectrum import SeaState
from wavesim.crestdistributions import rayleigh_pdf
from dataclasses import dataclass
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from scipy.integrate import quad


@dataclass
class weighted_cdf():
    """_computes the IS (weighted) ecdf for a given dataset and weights

    Args:
        dataset (np.ndarray): data
        weights (np.ndarray): weights for IS
    """

    dataset: np.ndarray
    weights: np.ndarray

    def evaluate(self, X: np.ndarray):
        """evaulate cdf at given points

        Args:
            X (np.ndarray): evaluation points
        """

        return np.sum((X[:, None] > self.dataset[None, :])*self.weights, axis=1)/np.sum(self.weights)

    __call__ = evaluate


@dataclass
class intr_krnl_cdf():
    """computes cdf by integration of a smoothed kernel density estimator of the pdf
    """

    kde: gaussian_kde

    def evaluate(self, X: np.ndarray):
        """evaluate the cdf at given points

        Args:
            X (np.ndarray): evaluation points

        Returns:
            np.ndarray: _description_
        """

        multi_X = np.vectorize(self.__single_X)

        return multi_X(X)

    __call__ = evaluate

    def __single_X(self, X: float):
        """integrate the pdf up to a single X point

        Args:
            X (float): integration upper limit

        Returns:
            _type_: _description_
        """

        return quad(self.kde, -5, X)[0]


@dataclass
class AbstractDistEst(ABC):
    """superclass for max sea state feature importance sampled distribution estimation

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
    def nz(self) -> int:
        """returns the number of z points to evaluate at

        Returns:
            int: number of z values
        """
        return len(self.z_values)

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
        f = rayleigh_pdf(self.cond_crests, self.sea_state.hs)
        self.weights = f/self.g
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

    def compute_cdf(self) -> None:
        """computes both versions of importance sampled distribution
        """

        if self.pdf is None:
            raise Exception("You must compute the pdf first")
        self.cdf = weighted_cdf(dataset=self.max_series, weights=self.weights)
        self.cdf_smooth = intr_krnl_cdf(self.pdf)

        return None

    def eval_cdf(self, X: np.ndarray, smooth: bool = True):
        """evaluates stored cdf and scales as needed

        Args:
            X (np.ndarray): evaluation points
            smooth (bool): if true then we use the kde integrand smoothed version of the cdf
        """

        if smooth:
            return self.cdf_smooth(X)**(self.sim_per_state*self.waves_per_sim)
        else:
            return self.cdf(X)**(self.sim_per_state*self.waves_per_sim)

    def compute_pdf(self) -> np.ndarray:
        """computes the pdf using a weighted kernel esimation method from the scipy package

        Args:
            X (np.ndarray): evaluation points
        """

        self.pdf = gaussian_kde(dataset=self.max_series, weights=self.weights, bw_method='scott')

        return None

    def eval_pdf(self, X: np.ndarray, smooth: bool = True) -> float:
        """evaluate and rescale the stored pdf estimate at specified points

        Args:
            X (np.ndarray): evaluation points
            smooth (bool): if true then we use the kde integrand smoothed version of the cdf

        Returns:
            float: estimated density at evaluation point
        """

        Q = self.sim_per_state * self.waves_per_sim

        if smooth:
            return self.pdf(X) * Q * self.cdf_smooth(X) ** (Q - 1)
        else:
            return self.pdf(X) * Q * self.cdf(X) ** (Q - 1)

    def plot_distribution(self, X: np.ndarray, log=True) -> None:
        """ plot the stored distribution

        Args:
            X (np.ndarray): evaluation points
            log (bool): boolean which decides if we plot cdf or log cdf
        """
        plt.figure()
        if log:
            plt.plot(X, np.log10(1-self.eval_cdf(X)))
            plt.xlabel('X')
            plt.ylabel('log10(1-p)')
            plt.title("Distribution of Sea State Max")
        else:
            plt.plot(X, self.eval_cdf(X))
            plt.xlabel('X')
            plt.ylabel('p')
            plt.title("Distribution of Sea State Max")
        plt.show()

    def plot_density(self, X) -> None:
        """plots the stored density
        """
        plt.figure()
        plt.plot(X, self.eval_pdf(X))
        plt.title("Density of Sea State Max")
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
class MorisonDistEst(AbstractDistEst):
    """sea-state max load distribution class

    Args:
        c_d (np.ndarray): drag coefficient array
        c_m (np.ndarray): inertia coefficient array
    """

    c_d: np.ndarray = 1
    c_m: np.ndarray = 1

    def compute_load(self) -> None:
        """compute loading from kinematics
        """

        self.load = MorisonLoad(self.kinematics, self.c_d, self.c_m)
        self.load.compute_load()

        return None

    def compute_sea_state_max(self) -> None:
        self.max_series = np.empty(self.sea_state.num_SS)
        for s in range(self.sea_state.num_SS):

            crests, _, _, _, _ = self.kinematics.retrieve_kinematics()
            crests = crests[:, s]
            load = self.load.retrieve_load()[:, s]
            # get maximums
            mins = argrelextrema(crests, np.less)[0]
            lower_min = np.max(mins[mins < self.kinematics.nt/2])
            upper_min = np.min(mins[mins > self.kinematics.nt/2])
            slice = load[lower_min:upper_min]
            self.max_series[s] = max(slice)

        return None
