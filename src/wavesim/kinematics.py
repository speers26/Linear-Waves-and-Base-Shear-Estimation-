'''
Code for generating kinematics, wave velocity, wave accelerations

'''

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import math
from wavesim.dispersion import alt_solve_dispersion, solve_dispersion, fDispersionSTOKES5
from wavesim.spectrum import AbstractSpectrum
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

# TODO: create classes for spatial waves


def spatial_random_wave(om_range: np.ndarray, phi_range: np.ndarray, Dr_spctrm: np.ndarray, t: np.ndarray,
                        x_range: np.ndarray, y_range: np.ndarray, h: float) -> np.ndarray:
    """returns random wave surface with frequency direction spectrum defined below

    Args:
        omega_range (np.ndarray): values of angular frequency to include
        phi_range (np.ndarray): values of direction to include
        t (np.ndarray): time (scalar)
        x_range (np.ndarray): range of x to evaluate over (forms a grid with y_range)
        y_range (np.ndarray): range of y to evaluate over (forms a grid with x_range)
        h (float): water depth [metres]

    Returns:
        eta (np.ndarray): random wave surface height [metres] (y_num, x_num)
    """
    np.random.seed(1452)

    om_num = len(om_range)
    phi_num = len(phi_range)
    x_num = len(x_range)
    y_num = len(y_range)
    d_om = om_range[1] - om_range[0]
    d_phi = phi_range[1] - phi_range[0]

    A = np.random.normal(0, 1, size=(phi_num, om_num)) * np.sqrt(Dr_spctrm * d_om * d_phi)
    B = np.random.normal(0, 1, size=(phi_num, om_num)) * np.sqrt(Dr_spctrm * d_om * d_phi)

    k = np.empty(om_num)
    for i_om, om in enumerate(om_range):
        k[i_om] = solve_dispersion(om, h, upp=1)

    eta = np.empty([y_num, x_num])

    for i_x, x in enumerate(x_range):
        for i_y, y in enumerate(y_range):
            k_x = np.outer(np.cos(phi_range), k)
            k_y = np.outer(np.sin(phi_range), k)
            om_t = np.tile(om_range * t, (phi_num, 1))
            eta[i_y, i_x] = np.sum(A * np.cos(k_x * x + k_y * y - om_t) + B * np.sin(k_x * x + k_y * y - om_t))

    return eta


@dataclass
class AbstractWaveKin(ABC):
    """ General wave kinematics class

    Args:
        t_values (np.ndarray): values in time at which to calculate kinematics [s]
        z_values (np.ndarray): depth values to calculate kimematics at [m]
    """

    sample_f: float
    period: float
    t_values: np.ndarray
    z_values: np.ndarray

    @property
    def t_values(self) -> np.ndarray:
        """computes time evaluation points from sample_frequency and time period

        Returns:
            np.ndarray: time evaluation points
        """

    @property
    def depth(self) -> float:
        """returns water depth as the minimun of z_values

        Returns:
            float: depth of sea bed
        """
        return -np.min(self.z_values)

    @property
    def nz(self) -> int:
        """retuns number of z values

        Returns:
            int: number of depth values to evaluate at
        """
        return len(self.z_values)

    @property
    def dz(self) -> float:
        """returns steps in z-values

        Returns:
            float: depth step size (homogenous)
        """
        return self.z_values[1] - self.z_values[0]

    @property
    def nt(self) -> int:
        """returns number of time points

        Returns:
            int: number of time points to evaluate at
        """
        return len(self.t_values)

    @property
    def zt_grid(self) -> np.ndarray:
        """returns grid for plotting over

        Returns:
            np.ndarray: meshgrid of depth and time values
        """
        return np.meshgrid(self.z_values, self.t_values)

    @abstractmethod
    def compute_kinematics(self) -> AbstractWaveKin:
        """compute kinematics for given time and z_values

        output stored in eta, u, w, du, dw

        Returns:
            AbstractWaveKin: returns self
        """

    def retrieve_kinematics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ outputs the kinematics stored in self.kinematics

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: crest height and kinematics
        """
        return self.eta, self.u, self.w, self.du, self.dw

    def plot_kinematics(self) -> None:
        """plots wave kinematics calculuated in compute_kinematics
        """
        plt.figure()
        plt.subplot(2, 2, 1)

        plt.scatter(self.zt_grid[1].flatten(), self.zt_grid[0].flatten(), s=1, c=self.u.flatten())
        plt.plot(self.t_values, self.eta, '-k')
        plt.title('u')
        plt.xlabel('time')
        plt.ylabel('depth')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.scatter(self.zt_grid[1].flatten(), self.zt_grid[0].flatten(), s=1, c=self.w.flatten())
        plt.plot(self.t_values, self.eta, '-k')
        plt.title('w')
        plt.xlabel('time')
        plt.ylabel('depth')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.scatter(self.zt_grid[1].flatten(), self.zt_grid[0].flatten(), s=1, c=self.du.flatten())
        plt.plot(self.t_values, self.eta, '-k')
        plt.title('du')
        plt.xlabel('time')
        plt.ylabel('depth')
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.scatter(self.zt_grid[1].flatten(), self.zt_grid[0].flatten(), s=1, c=self.dw.flatten())
        plt.plot(self.t_values, self.eta, '-k')
        plt.title('dw')
        plt.xlabel('time')
        plt.ylabel('depth')
        plt.colorbar()

        plt.show()


@dataclass
class LinearKin(AbstractWaveKin):
    """ Linear Random Wave Kinematics Class

    Args:
        spctr (Spectrum): desired spectral density of the wave surface
    """

    spctr: AbstractSpectrum

    def compute_kinematics(self, cond: bool, a: float = 0, NewWave: bool = False) -> LinearKin:
        """computes linear wave kinematics

        Args:
            cond (bool): Set to True to generate a conditioned wave series
            a (float, optional): Conditioned crest elevation at t=0. Defaults to 0.
            NewWave (bool, optional): Set to True to generate a NewWave. Defaults to False.

        Returns:
            LinearKin: returns self
        """
        if NewWave:
            A = np.zeros(shape=(1, self.spctr.nf))
            B = np.zeros(shape=(1, self.spctr.nf))

        else:
            A = np.random.normal(0, 1, size=(1, self.spctr.nf)) * np.sqrt(self.spctr.density*self.spctr.df)
            B = np.random.normal(0, 1, size=(1, self.spctr.nf)) * np.sqrt(self.spctr.density*self.spctr.df)

        if cond:
            m = 0

            c = self.spctr.df * self.spctr.density
            d = self.spctr.df * self.spctr.density * self.spctr.omega

            Q = (a - np.sum(A))/np.sum(c)
            R = (m - np.sum(self.spctr.omega * B))/np.sum(d*self.spctr.omega)

            A = A + Q * c
            B = B + R * d

        i = complex(0, 1)
        g1 = A + B * i

        self.eta = np.real(fftshift(fft(g1)))[0]

        k = alt_solve_dispersion(self.spctr.omega, self.depth)

        self.u = np.empty((self.spctr.nf, len(self.z_values)))
        self.du = np.empty((self.spctr.nf, len(self.z_values)))
        self.w = np.empty((self.spctr.nf, len(self.z_values)))
        self.dw = np.empty((self.spctr.nf, len(self.z_values)))

        d = self.depth
        for i_z, z in enumerate(self.z_values):

            z_init = z
            if z > -1:
                z = -1

            qf1 = (np.cosh(k*(z+d))) / (np.sinh(k*d))
            qf2 = (np.sinh(k*(z+d))) / (np.sinh(k*d))

            qf1[[i for i in range(len(qf1)) if math.isnan(qf1[i])]] = 1
            qf2[[i for i in range(len(qf2)) if math.isnan(qf2[i])]] = 1

            g2 = (A+B*i) * 2*np.pi*self.spctr.frequency * qf1
            g3 = (B-A*i) * (2*np.pi*self.spctr.frequency)**2 * qf1
            g4 = (B-A*i) * (2*np.pi*self.spctr.frequency) * qf2
            g5 = (-A-B*i) * (2*np.pi*self.spctr.frequency)**2 * qf2

            self.u[:, i_z] = np.real(fftshift(fft(g2))) * (z_init < self.eta)  # * (z_init < 0)
            self.du[:, i_z] = np.real(fftshift(fft(g3))) * (z_init < self.eta)  # * (z_init < 0)
            self.w[:, i_z] = np.real(fftshift(fft(g4))) * (z_init < self.eta)  # * (z_init < 0)
            self.dw[:, i_z] = np.real(fftshift(fft(g5))) * (z_init < self.eta)  # * (z_init < 0)

        return self


@dataclass
class DetWaveKin(AbstractWaveKin):
    """ deterministic wave sim class

    Args:
        H (np.ndarray): wave height [m]
        T (np.ndarray): wave period [s]
        g (np.ndarray): acc due to gravity [ms^-2]
        x (np.ndarray): point in space to evaluate kimematics at [m]
        theta (np.ndarray): angle of propogation of the wave [radians]

    """
    H: np.ndarray
    T: np.ndarray
    g: float = 9.81
    x: np.ndarray = 0  # always set this to 0 for now TODO: implement this for all waves
    theta: np.ndarray = 0  # always set to 0 for now TODO: implement this for all waves

    @property
    def omega(self) -> float:
        """returns the angular freq

        Returns:
            float: peak angular frequency
        """
        return 2*np.pi/self.T


@dataclass
class AiryKin(DetWaveKin):
    """ Airy kinematics class
    """
    @property
    def k(self) -> float:
        """returns wave number for kinematics calculation

        Returns:
            float: wave number
        """
        return alt_solve_dispersion(self.omega, self.depth)

    def compute_kinematics(self) -> AiryKin:
        """computes airy wave kinematics and stores them in self

        Returns:
            AiryKin: returns self
        """
        self.eta = np.empty(self.nt)
        self.u = np.empty((self.nt, self.nz))
        self.w = np.empty((self.nt, self.nz))
        self.du = np.empty((self.nt, self.nz))
        self.dw = np.empty((self.nt, self.nz))

        for i_t, t in enumerate(self.t_values):
            for i_z, z in enumerate(self.z_values):

                A = self.H / 2

                self.eta[i_t] = A * np.sin(self.omega * t - self.k * self.x)

                if z > self.eta[i_t]:
                    self.u[i_t, i_z] = self.w[i_t, i_z] = self.du[i_t, i_z] = self.dw[i_t, i_z] = 0

                else:
                    self.u[i_t, i_z] = self.omega * A * ((np.cosh(self.k * (self.depth + z))) /
                                                         (np.sinh(self.k * self.depth))) \
                                                        * np.sin(self.omega * t - self.k * self.x)

                    self.w[i_t, i_z] = self.omega * A * ((np.sinh(self.k * (self.depth + z))) /
                                                         (np.sinh(self.k * self.depth))) \
                                                      * np.cos(self.omega * t - self.k * self.x)

                    self.du[i_t, i_z] = self.omega ** 2 * A * ((np.cosh(self.k * (self.depth + z))) /
                                                               (np.sinh(self.k * self.depth))) \
                                                            * np.cos(self.omega * t - self.k * self.x)

                    self.dw[i_t, i_z] = -self.omega ** 2 * A * ((np.sinh(self.k * (self.depth + z))) /
                                                                (np.sinh(self.k * self.depth))) \
                                                             * np.sin(self.omega * t - self.k * self.x)

        return self


@dataclass
class StokesKin(DetWaveKin):
    """ Stokes kinematics class
    """

    @property
    def k(self) -> float:
        """return k for stokes wave

        Returns:
            float: wave number
        """
        return fDispersionSTOKES5(self.depth, self.H, self.omega)

    def compute_kinematics(self) -> StokesKin:
        """ computes Stokes wave kinematics and stores them in self

        Returns:
            StokesKin: returns self
        """
        self.eta = np.empty(self.nt)
        self.u = np.empty((self.nt, self.nz))
        self.w = np.empty((self.nt, self.nz))
        self.du = np.empty((self.nt, self.nz))
        self.dw = np.empty((self.nt, self.nz))

        for i_t, t in enumerate(self.t_values):
            for i_z, z in enumerate(self.z_values):

                kd = self.k * self.depth
                # !e Initialisation
                S = 1 / np.cosh(2 * kd)
                # Calculation of the A coefficients
                Aco = np.empty(9)
                Aco[0] = 1 / np.sinh(kd)
                Aco[1] = 3 * (S ** 2) / (2 * ((1. - S) ** 2))
                Aco[2] = (-4 - 20 * S + 10 * (S ** 2) - 13 * (S ** 3)) / (8 * np.sinh(kd) * ((1 - S) ** 3))
                Aco[3] = (-2 * (S ** 2) + 11 * (S ** 3)) / (8 * np.sinh(kd) * ((1.-S) ** 3))
                Aco[4] = (12 * S - 14 * (S ** 2) - 264 * (S ** 3) - 45 * (S ** 4) - 13 * (S ** 5)) / (24*((1.-S)**5))
                Aco[5] = (10 * (S ** 3) - 174 * (S ** 4) + 291 * (S ** 5) + 278 * (S ** 6)) / (48 * (3 + 2 * S)
                                                                                               * ((1 - S) ** 5))
                Aco[6] = (-1184 + 32 * S + 13232 * (S ** 2) + 21712 * (S ** 3) + 20940 * (S ** 4) + 12554 * (S ** 5)
                          - 500 * (S ** 6) - 3341 * (S ** 7) - 670 * (S ** 8)) / (64 * np.sinh(kd) * (3 + 2 * S)
                                                                                  * (4 + S) * ((1 - S) ** 6))
                Aco[7] = (4 * S + 105 * (S ** 2) + 198 * (S ** 3) - 1376 * (S ** 4) - 1302 * (S ** 5) - 117 * (S ** 6) +
                          58 * (S ** 7))/(32 * np.sinh(kd) * (3 + 2 * S) * ((1 - S) ** 6))
                Aco[8] = (-6 * (S ** 3) + 272 * (S ** 4) - 1552 * (S ** 5) + 852 * (S ** 6) + 2029 * (S ** 7) + 430
                          * (S ** 8)) / (64 * np.sinh(kd) * (3 + 2 * S) * (4 + S) * ((1 - S) ** 6))
                # Calculation of the B coefficients
                Bco = np.empty(6)
                Bco[0] = (1 / np.tanh(kd)) * (1 + 2 * S) / (2 * (1 - S))
                Bco[1] = -3 * (1 + 3 * S + 3 * (S ** 2) + 2 * (S ** 3)) / (8 * ((1 - S) ** 3))
                Bco[2] = (1 / np.tanh(kd)) * (6 - 26 * S - 182 * (S ** 2) - 204 * (S ** 3) - 25 * (S ** 4) + 26
                                              * (S ** 5)) / (6 * (3 + 2 * S) * ((1 - S) ** 4))
                Bco[3] = (1./np.tanh(kd)) * (24 + 92 * S + 122 * (S ** 2) + 66 * (S ** 3) + 67 * (S ** 4) + 34
                                             * (S ** 5)) / (24 * (3 + 2 * S) * ((1 - S) ** 4))
                Bco[4] = 9 * (132 + 17 * S - 2216 * (S ** 2) - 5897 * (S ** 3) - 6292 * (S ** 4) - 2687 * (S ** 5)
                              + 194 * (S ** 6) + 467 * (S ** 7) + 82 * (S ** 8)) / (128 * (3 + 2 * S) * (4 + S)
                                                                                    * ((1 - S) ** 6))
                Bco[5] = 5 * (300 + 1579 * S + 3176 * (S ** 2) + 2949 * (S ** 3) + 1188 * (S ** 4) + 675 * (S ** 5)
                              + 1326 * (S ** 6) + 827 * (S ** 7) + 130 * (S ** 8)) / (384 * (3 + 2 * S) * (4 + S)
                                                                                      * ((1 - S) ** 6))
                # Calculation of the C coefficients
                Cco = np.empty(3)
                Cco[0] = np.sqrt(np.tanh(kd))
                Cco[1] = (np.sqrt(np.tanh(kd)) * (2 + 7 * S ** 2)) / (4 * (1-S) ** 2)
                Cco[2] = (np.sqrt(np.tanh(kd)) * (4 + 32 * S - 116 * S ** 2 - 400 * S ** 3 - 71 * S ** 4 + 146
                                                  * S ** 5)) / (32 * (1 - S) ** 5)
                # Calculation of the D coefficients
                Dco = np.empty(2)
                Dco[0] = -0.5 * np.sqrt(1 / np.tanh(kd))
                Dco[1] = (np.sqrt(1 / np.tanh(kd)) * (2 + 4 * S + S ** 2 + 2 * S ** 3)) / (8 * (1 - S) ** 3)
                # Calculation of the E coefficients
                Eco = np.empty(2)
                Eco[0] = (np.tanh(kd) * (2 + 2 * S + 5 * S ** 2)) / (4 * (1 - S) ** 2)
                Eco[1] = (np.tanh(kd) * (8 + 12 * S - 152 * S ** 2 - 308 * S ** 3 - 42 * S ** 4 + 77 * S ** 5)) \
                    / (32 * (1 - S) ** 5)

                # calculate properties
                # Initialising coefficients
                A11 = Aco[0]
                A22 = Aco[1]
                A31 = Aco[2]
                A33 = Aco[3]
                A42 = Aco[4]
                A44 = Aco[5]
                A51 = Aco[6]
                A53 = Aco[7]
                A55 = Aco[8]
                B22 = Bco[0]
                B31 = Bco[1]
                B42 = Bco[2]
                B44 = Bco[3]
                B53 = Bco[4]
                B55 = Bco[5]
                C0 = Cco[0]
                # Wave steepness
                epsilon = self.H/2 * self.k
                #
                psi = self.k * self.x - self.omega * t

                k_z_plus_h = self.k * (z + self.depth)
                # z
                self.eta[i_t] = (1 / self.k) * (epsilon * np.cos(psi) + B22 * (epsilon ** 2) * np.cos(2 * psi)
                                                + B31 * (epsilon ** 3) * (np.cos(psi) - np.cos(3 * psi))
                                                + (epsilon ** 4) * (B42 * np.cos(2 * psi) + B44 * np.cos(4 * psi))
                                                + (epsilon ** 5) * (-(B53 + B55) * np.cos(psi) + B53 * np.cos(3 * psi)
                                                + B55 * np.cos(5 * psi)))

                if z > self.eta[i_t]:
                    self.u[i_t, i_z] = self.w[i_t, i_z] = self.du[i_t, i_z] = self.dw[i_t, i_z] = 0

                else:
                    # u calculation
                    self.u[i_t, i_z] = (C0 * np.sqrt(self.g / self.k ** 3)) * (self.k * np.cos(self.theta)) \
                        * (A11 * epsilon * np.cosh(k_z_plus_h) * np.cos(psi)
                            + A22 * (epsilon ** 2) * np.cosh(2 * k_z_plus_h) * 2 * np.cos(2 * psi)
                            + A31 * (epsilon ** 3) * np.cosh(k_z_plus_h) * np.cos(psi)
                            + A33 * (epsilon ** 3) * np.cosh(3 * k_z_plus_h) * 3 * np.cos(3 * psi)
                            + A42 * (epsilon ** 4) * np.cosh(2 * k_z_plus_h) * 2 * np.cos(2 * psi)
                            + A44 * (epsilon ** 4) * np.cosh(4 * k_z_plus_h) * 4 * np.cos(4 * psi)
                            + A51 * (epsilon ** 5) * np.cosh(k_z_plus_h) * np.cos(psi)
                            + A53 * (epsilon ** 5) * np.cosh(3 * k_z_plus_h) * 3 * np.cos(3 * psi)
                            + A55 * (epsilon ** 5) * np.cosh(5 * k_z_plus_h) * 5 * np.cos(5 * psi))
                    # w calculation
                    self.w[i_t, i_z] = (C0 * np.sqrt(self.g / self.k ** 3)) * self.k \
                        * (A11 * epsilon * np.sinh(k_z_plus_h)*np.sin(psi)
                            + A22 * (epsilon ** 2) * np.sinh(2 * k_z_plus_h) * 2 * np.sin(2 * psi)
                            + A31 * (epsilon ** 3) * np.sinh(k_z_plus_h) * np.sin(psi)
                            + A33 * (epsilon ** 3) * np.sinh(3 * k_z_plus_h) * 3 * np.sin(3 * psi)
                            + A42 * (epsilon ** 4) * np.sinh(2 * k_z_plus_h) * 2 * np.sin(2 * psi)
                            + A44 * (epsilon ** 4) * np.sinh(4 * k_z_plus_h) * 4 * np.sin(4 * psi)
                            + A51 * (epsilon ** 5) * np.sinh(k_z_plus_h) * np.sin(psi)
                            + A53 * (epsilon ** 5) * np.sinh(3 * k_z_plus_h) * 3 * np.sin(3 * psi)
                            + A55 * (epsilon ** 5) * np.sinh(5 * k_z_plus_h) * 5 * np.sin(5 * psi))
                    #  dudt horizontal acceleration
                    self.du[i_t, i_z] = (C0 * np.sqrt(self.g / self.k ** 3)) * (self.k * np.cos(self.theta)) \
                        * (A11 * (epsilon) * np.cosh(k_z_plus_h) * self.omega * np.sin(psi)
                            + A22 * (epsilon ** 2) * np.cosh(2 * k_z_plus_h) * 2 * self.omega * np.sin(2 * psi)
                            + A31 * (epsilon ** 3) * np.cosh(k_z_plus_h) * self.omega * np.sin(psi)
                            + A33 * (epsilon ** 3) * np.cosh(3 * k_z_plus_h) * 3 * self.omega * np.sin(3 * psi)
                            + A42 * (epsilon ** 4) * np.cosh(2 * k_z_plus_h) * 2 * self.omega * np.sin(2 * psi)
                            + A44 * (epsilon ** 4) * np.cosh(4 * k_z_plus_h) * 4 * self.omega * np.sin(4 * psi)
                            + A51 * (epsilon ** 5) * np.cosh(k_z_plus_h) * self.omega * np.sin(psi)
                            + A53 * (epsilon ** 5) * np.cosh(3 * k_z_plus_h) * 3 * self.omega * np.sin(3 * psi)
                            + A55 * (epsilon ** 5) * np.cosh(5 * k_z_plus_h) * 5 * self.omega * np.sin(5 * psi))
                    # dwdt vertical acceleration
                    self.dw[i_t, i_z] = (C0 * np.sqrt(self.g / self.k ** 3)) * self.k \
                        * (A11 * epsilon * np.sinh(k_z_plus_h)*self.omega*-np.cos(psi)
                            + A22 * (epsilon ** 2) * np.sinh(2 * k_z_plus_h) * 2 * self.omega * -np.cos(2 * psi)
                            + A31 * (epsilon ** 3) * np.sinh(k_z_plus_h) * self.omega * -np.cos(psi)
                            + A33 * (epsilon ** 3) * np.sinh(3 * k_z_plus_h) * 3 * self.omega * -np.cos(3 * psi)
                            + A42 * (epsilon ** 4) * np.sinh(2 * k_z_plus_h) * 2 * self.omega * -np.cos(2 * psi)
                            + A44 * (epsilon ** 4) * np.sinh(4 * k_z_plus_h) * 4 * self.omega * -np.cos(4 * psi)
                            + A51 * (epsilon ** 5) * np.sinh(k_z_plus_h) * self.omega * - np.cos(psi)
                            + A53 * (epsilon ** 5) * np.sinh(3 * k_z_plus_h) * 3 * self.omega * -np.cos(3 * psi)
                            + A55 * (epsilon ** 5) * np.sinh(5 * k_z_plus_h) * 5 * self.omega * -np.cos(5 * psi))

                if z > self.eta[i_t]:
                    self.u[i_t, i_z] = self.w[i_t, i_z] = self.du[i_t, i_z] = self.dw[i_t, i_z] = 0
        return self
