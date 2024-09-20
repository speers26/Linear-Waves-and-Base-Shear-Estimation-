'''
Code for generating kinematics, wave velocity, wave accelerations

'''

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import math
from wavesim.dispersion import alt_solve_dispersion, solve_dispersion, fDispersionSTOKES5
from wavesim.spectrum import AbstractSpectrum, SeaState
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt
import warnings

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


# Directional Spectrum

# nT = [same as time version] nTheta = 32 Theta = np.linspace(from = 0, by=11.25, to=348.75)

# Defines S = [nT, nTheta]

# S wrapped normal distribution in Theta dimension producted with JONSWAP in Freq/Dimension (Code from Jake)

# A, B = randn( nT, nTheta) * S * dtheta * dt

# This is bigger than time only e.g. nTheta=1

# Adjust A, B as time case for conditional waves (no change)

# Z = A + iB

# k_x = cos()/sin stuff

# k = k_x * x + k_y * y

# Z = sum(exp(1i * k) .* Z,2)

# eta = fftshift(real(fft(Z,Spec.nf,1)),1

@dataclass 
class SpatialLinearKin():
    """_summary_

    Returns:
        _type_: _description_
    """

    sample_f: float
    period: float
    x_values: np.ndarray
    y_values: np.ndarray
    z_values: np.ndarray
    hs: float
    tp: float
    theta_p: float

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
    def ny(self) -> int:
        """retuns number of y values

        Returns:
            int: number of y values to evaluate at
        """
        return len(self.y_values)

    @property
    def dy(self) -> float:
        """returns steps in y-values

        Returns:
            float: y step size (homogenous)
        """
        return self.y_values[1] - self.y_values[0]

    @property
    def nx(self) -> int:
        """retuns number of x values

        Returns:
            int: number of x values to evaluate at
        """
        return len(self.x_values)

    @property
    def dx(self) -> float:
        """returns steps in x-values

        Returns:
            float: x step size (homogenous)
        """
        return self.x_values[1] - self.x_values[0]

    @property
    def nt(self) -> int:
        """returns number of time points

        Returns:
            int: number of time points to evaluate at
        """
        return int(np.floor(self.period*self.sample_f))  # number of time points to evaluate

    @property
    def t_values(self) -> np.ndarray:
        """returns the t_values to evaluate at

        Returns:
            np.ndarray: t_values
        """
        dt = 1/self.sample_f
        return np.linspace(-self.nt/2, self.nt/2 - 1, int(self.nt)) * dt  # centering time around 0


@dataclass
class AbstractWaveKin(ABC):
    """ General wave kinematics class

    Args:
        t_values (np.ndarray): values in time at which to calculate kinematics [s]
        z_values (np.ndarray): depth values to calculate kimematics at [m]
    """

    sample_f: float
    period: float
    z_values: np.ndarray
    sea_state: SeaState

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
        return int(np.floor(self.period*self.sample_f))  # number of time points to evaluate

    @property
    def t_values(self) -> np.ndarray:
        """returns the t_values to evaluate at

        Returns:
            np.ndarray: t_values
        """
        dt = 1/self.sample_f
        return np.linspace(-self.nt/2, self.nt/2 - 1, int(self.nt)) * dt  # centering time around 0

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

    def plot_kinematics(self, s: int = 0) -> None:
        """plots wave kinematics calculuated in compute_kinematics

        Args:
            s (int): index of sea state to plot

        """
        plt.figure()
        plt.subplot(2, 2, 1)

        plt.scatter(self.zt_grid[1].flatten(), self.zt_grid[0].flatten(), s=1, c=self.u[:, :, s].flatten())
        plt.plot(self.t_values, self.eta[:, s], '-k')
        plt.title('u')
        plt.xlabel('time')
        plt.ylabel('depth')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        plt.scatter(self.zt_grid[1].flatten(), self.zt_grid[0].flatten(), s=1, c=self.w[:, :, s].flatten())
        plt.plot(self.t_values, self.eta[:, s], '-k')
        plt.title('w')
        plt.xlabel('time')
        plt.ylabel('depth')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.scatter(self.zt_grid[1].flatten(), self.zt_grid[0].flatten(), s=1, c=self.du[:, :, s].flatten())
        plt.plot(self.t_values, self.eta[:, s], '-k')
        plt.title('du')
        plt.xlabel('time')
        plt.ylabel('depth')
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.scatter(self.zt_grid[1].flatten(), self.zt_grid[0].flatten(), s=1, c=self.dw[:, :, s].flatten())
        plt.plot(self.t_values, self.eta[:, s], '-k')
        plt.title('dw')
        plt.xlabel('time')
        plt.ylabel('depth')
        plt.colorbar()

        plt.show()


@dataclass
class LinearKin(AbstractWaveKin):
    """ Linear Random Wave Kinematics Class

    """

    @property
    def frequency(self) -> np.ndarray:
        """calculates and returns the frequencies to evaluate the spectrum at for given period and sample f

        Returns:
            np.ndarray: contributing frequencies
        """

        f_range = np.linspace(1e-3, self.nt - 1, int(self.nt)) / (self.nt / self.sample_f)  # selecting frequency

        return f_range

    def compute_spectrum(self) -> AbstractSpectrum:
        """computes the spectral densities

        Returns:
            LinearKin: returns self
        """
        self.spctr = []
        for s in range(self.sea_state.num_SS):
            spctr = self.sea_state.spctr_type(self.sea_state.hs[s], self.sea_state.tp[s], self.frequency)
            spctr.compute_density()
            spctr.compute_omega_density()
            self.spctr.append(spctr)
        return self

    def compute_kinematics(self, cond: bool, a: np.ndarray = 0, NewWave: bool = False) -> LinearKin:
        """computes linear wave kinematics

        Args:
            cond (bool): Set to True to generate a conditioned wave series
            a (np.ndarray, optional): Conditioned crest elevation at t=0. Defaults to 0.
            NewWave (bool, optional): Set to True to generate a NewWave. Defaults to False.

        Returns:
            LinearKin: returns self
        """

        self.eta = np.empty((self.spctr[0].nf, self.sea_state.num_SS))
        self.u = np.empty((self.spctr[0].nf, len(self.z_values), self.sea_state.num_SS))
        self.du = np.empty((self.spctr[0].nf, len(self.z_values), self.sea_state.num_SS))
        self.w = np.empty((self.spctr[0].nf, len(self.z_values), self.sea_state.num_SS))
        self.dw = np.empty((self.spctr[0].nf, len(self.z_values), self.sea_state.num_SS))

        for s in range(self.sea_state.num_SS):

            if NewWave:
                A = np.zeros(shape=(1, self.spctr[s].nf))
                B = np.zeros(shape=(1, self.spctr[s].nf))

            else:
                A = np.random.normal(0, 1, size=(1, self.spctr[s].nf)) * np.sqrt(self.spctr[s].density*self.spctr[s].df)
                B = np.random.normal(0, 1, size=(1, self.spctr[s].nf)) * np.sqrt(self.spctr[s].density*self.spctr[s].df)

            if cond:
                m = 0

                c = self.spctr[s].df * self.spctr[s].density
                d = self.spctr[s].df * self.spctr[s].density * self.spctr[s].omega

                Q = (a[s] - np.sum(A))/np.sum(c)
                R = (m - np.sum(self.spctr[s].omega * B))/np.sum(d*self.spctr[s].omega)

                A = A + Q * c
                B = B + R * d

            i = complex(0, 1)
            g1 = A + B * i

            self.eta[:, s] = np.real(fftshift(fft(g1)))[0]

            k = alt_solve_dispersion(self.spctr[s].omega, self.depth)

            d = self.depth
            for i_z, z in enumerate(self.z_values):

                z_init = z
                if z > -1:
                    z = -1

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    qf1 = (np.cosh(k*(z+d))) / (np.sinh(k*d))
                    qf2 = (np.sinh(k*(z+d))) / (np.sinh(k*d))

                qf1[[i for i in range(len(qf1)) if math.isnan(qf1[i])]] = 1
                qf2[[i for i in range(len(qf2)) if math.isnan(qf2[i])]] = 1

                g2 = (A+B*i) * 2*np.pi*self.spctr[s].frequency * qf1
                g3 = (B-A*i) * (2*np.pi*self.spctr[s].frequency)**2 * qf1
                g4 = (B-A*i) * (2*np.pi*self.spctr[s].frequency) * qf2
                g5 = (-A-B*i) * (2*np.pi*self.spctr[s].frequency)**2 * qf2

                self.u[:, i_z, s] = np.real(fftshift(fft(g2))) * (z_init < self.eta[:, s]) + np.cos(self.sea_state.current_incidence) * self.sea_state.current
                self.du[:, i_z, s] = np.real(fftshift(fft(g3))) * (z_init < self.eta[:, s])
                self.w[:, i_z, s] = np.real(fftshift(fft(g4))) * (z_init < self.eta[:, s])
                self.dw[:, i_z, s] = np.real(fftshift(fft(g5))) * (z_init < self.eta[:, s])

        return self


@dataclass
class AiryKin(AbstractWaveKin):
    """ Airy kinematics class

    Args:
        x (np.ndarray): point in space to evaluate kimematics at [m] (placeholder)
    """

    x: np.ndarray = 0  # always set this to 0 for now TODO: implement this for all waves

    @property
    def k(self) -> float:
        """returns wave number for kinematics calculation

        Returns:
            float: wave number
        """
        return alt_solve_dispersion(self.sea_state.omega_det, self.depth)

    def compute_kinematics(self) -> AiryKin:
        """computes airy wave kinematics and stores them in self

        Returns:
            AiryKin: returns self
        """
        self.eta = np.empty((self.nt, self.sea_state.num_SS))
        self.u = np.empty((self.nt, self.nz, self.sea_state.num_SS))
        self.w = np.empty((self.nt, self.nz, self.sea_state.num_SS))
        self.du = np.empty((self.nt, self.nz, self.sea_state.num_SS))
        self.dw = np.empty((self.nt, self.nz, self.sea_state.num_SS))

        for s in range(self.sea_state.num_SS):
            for i_t, t in enumerate(self.t_values):
                for i_z, z in enumerate(self.z_values):

                    A = self.sea_state.H_det[s] / 2

                    self.eta[i_t, s] = A * np.sin(self.sea_state.omega_det[s] * t - self.k[s] * self.x)

                    if z > self.eta[i_t, s]:
                        self.u[i_t, i_z, s] = self.w[i_t, i_z, s] = self.du[i_t, i_z, s] = self.dw[i_t, i_z, s] = 0

                    else:
                        self.u[i_t, i_z, s] = self.sea_state.omega_det[s] * A * ((np.cosh(self.k[s] * (self.depth + z)))
                                                                                 / (np.sinh(self.k[s] * self.depth))) \
                            * np.sin(self.sea_state.omega_det[s] * t - self.k[s] * self.x)

                        self.w[i_t, i_z, s] = self.sea_state.omega_det[s] * A * ((np.sinh(self.k[s] * (self.depth + z)))
                                                                                 / (np.sinh(self.k[s] * self.depth))) \
                            * np.cos(self.sea_state.omega_det[s] * t - self.k[s] * self.x)

                        self.du[i_t, i_z, s] = self.sea_state.omega_det[s] ** 2 * A * \
                            ((np.cosh(self.k[s] * (self.depth + z))) / (np.sinh(self.k[s] * self.depth))) \
                            * np.cos(self.sea_state.omega_det[s] * t - self.k[s] * self.x)

                        self.dw[i_t, i_z, s] = -self.sea_state.omega_det[s] ** 2 * A * \
                            ((np.sinh(self.k[s] * (self.depth + z))) / (np.sinh(self.k[s] * self.depth))) \
                            * np.sin(self.sea_state.omega_det[s] * t - self.k[s] * self.x)

        return self


@dataclass
class StokesKin(AbstractWaveKin):
    """ Stokes kinematics class

        Args:
        x (np.ndarray): point in space to evaluate kimematics at [m] (placeholder)
    """

    x: np.ndarray = 0  # always set this to 0 for now TODO: implement this for all waves

    @property
    def k(self) -> float:
        """return k for stokes wave

        Returns:
            float: wave number
        """
        # TODO: vectorise this
        k = np.empty(self.sea_state.num_SS)
        for s in range(self.sea_state.num_SS):
            k[s] = fDispersionSTOKES5(self.depth, self.sea_state.H_det[s], self.sea_state.omega_det[s])

        return k

    def compute_kinematics(self) -> StokesKin:
        """ computes Stokes wave kinematics and stores them in self

        Returns:
            StokesKin: returns self
        """
        self.eta = np.empty((self.nt, self.sea_state.num_SS))
        self.u = np.empty((self.nt, self.nz, self.sea_state.num_SS))
        self.w = np.empty((self.nt, self.nz, self.sea_state.num_SS))
        self.du = np.empty((self.nt, self.nz, self.sea_state.num_SS))
        self.dw = np.empty((self.nt, self.nz, self.sea_state.num_SS))

        for s in range(self.sea_state.num_SS):
            for i_t, t in enumerate(self.t_values):
                for i_z, z in enumerate(self.z_values):

                    kd = self.k[s] * self.depth
                    # !e Initialisation
                    S = 1 / np.cosh(2 * kd)
                    # Calculation of the A coefficients
                    Aco = np.empty(9)
                    Aco[0] = 1 / np.sinh(kd)
                    Aco[1] = 3 * (S ** 2) / (2 * ((1. - S) ** 2))
                    Aco[2] = (-4 - 20 * S + 10 * (S ** 2) - 13 * (S ** 3)) / (8 * np.sinh(kd) * ((1 - S) ** 3))
                    Aco[3] = (-2 * (S ** 2) + 11 * (S ** 3)) / (8 * np.sinh(kd) * ((1.-S) ** 3))
                    Aco[4] = (12 * S - 14 * (S ** 2) - 264 * (S ** 3) - 45 * (S ** 4) - 13
                              * (S ** 5)) / (24*((1.-S)**5))
                    Aco[5] = (10 * (S ** 3) - 174 * (S ** 4) + 291 * (S ** 5) + 278 * (S ** 6)) / (48 * (3 + 2 * S)
                                                                                                   * ((1 - S) ** 5))
                    Aco[6] = (-1184 + 32 * S + 13232 * (S ** 2) + 21712 * (S ** 3) + 20940 * (S ** 4) + 12554 * (S ** 5)
                              - 500 * (S ** 6) - 3341 * (S ** 7) - 670 * (S ** 8)) / (64 * np.sinh(kd) * (3 + 2 * S)
                                                                                      * (4 + S) * ((1 - S) ** 6))
                    Aco[7] = (4 * S + 105 * (S ** 2) + 198 * (S ** 3) - 1376 * (S ** 4) - 1302 * (S ** 5) - 117
                              * (S ** 6) + 58 * (S ** 7))/(32 * np.sinh(kd) * (3 + 2 * S) * ((1 - S) ** 6))
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
                    epsilon = self.sea_state.H_det[s]/2 * self.k[s]
                    #
                    psi = self.k[s] * self.x - self.sea_state.omega_det[s] * t

                    k_z_plus_h = self.k[s] * (z + self.depth)
                    # z
                    self.eta[i_t, s] = (1 / self.k[s]) * (epsilon * np.cos(psi) + B22 * (epsilon ** 2) * np.cos(2 * psi)
                                                          + B31 * (epsilon ** 3) * (np.cos(psi) - np.cos(3 * psi))
                                                          + (epsilon ** 4) * (B42 * np.cos(2 * psi) + B44
                                                                              * np.cos(4 * psi))
                                                          + (epsilon ** 5) * (-(B53 + B55) * np.cos(psi) + B53
                                                                              * np.cos(3 * psi)
                                                          + B55 * np.cos(5 * psi)))

                    if z > self.eta[i_t, s]:
                        self.u[i_t, i_z, s] = self.w[i_t, i_z, s] = self.du[i_t, i_z, s] = self.dw[i_t, i_z, s] = 0

                    else:
                        # u calculation
                        self.u[i_t, i_z, s] = (C0 * np.sqrt(self.sea_state.g / self.k[s] ** 3)) \
                            * (self.k[s] * np.cos(self.sea_state.theta)) \
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
                        self.w[i_t, i_z, s] = (C0 * np.sqrt(self.sea_state.g / self.k[s] ** 3)) * self.k[s] \
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
                        self.du[i_t, i_z, s] = (C0 * np.sqrt(self.sea_state.g / self.k[s] ** 3)) \
                            * (self.k[s] * np.cos(self.sea_state.theta)) \
                            * (A11 * (epsilon) * np.cosh(k_z_plus_h) * self.sea_state.omega_det[s] * np.sin(psi)
                                + A22 * (epsilon ** 2) * np.cosh(2 * k_z_plus_h) * 2 * self.sea_state.omega_det[s]
                                * np.sin(2 * psi)
                                + A31 * (epsilon ** 3) * np.cosh(k_z_plus_h) * self.sea_state.omega_det[s] * np.sin(psi)
                                + A33 * (epsilon ** 3) * np.cosh(3 * k_z_plus_h) * 3 * self.sea_state.omega_det[s]
                                * np.sin(3 * psi)
                                + A42 * (epsilon ** 4) * np.cosh(2 * k_z_plus_h) * 2 * self.sea_state.omega_det[s]
                                * np.sin(2 * psi)
                                + A44 * (epsilon ** 4) * np.cosh(4 * k_z_plus_h) * 4 * self.sea_state.omega_det[s]
                                * np.sin(4 * psi)
                                + A51 * (epsilon ** 5) * np.cosh(k_z_plus_h) * self.sea_state.omega_det[s] * np.sin(psi)
                                + A53 * (epsilon ** 5) * np.cosh(3 * k_z_plus_h) * 3 * self.sea_state.omega_det[s]
                                * np.sin(3 * psi)
                                + A55 * (epsilon ** 5) * np.cosh(5 * k_z_plus_h) * 5 * self.sea_state.omega_det[s]
                                * np.sin(5 * psi))
                        # dwdt vertical acceleration
                        self.dw[i_t, i_z, s] = (C0 * np.sqrt(self.sea_state.g / self.k[s] ** 3)) * self.k[s] \
                            * (A11 * epsilon * np.sinh(k_z_plus_h)*self.sea_state.omega_det[s]*-np.cos(psi)
                                + A22 * (epsilon ** 2) * np.sinh(2 * k_z_plus_h) * 2 * self.sea_state.omega_det[s]
                                * -np.cos(2 * psi)
                                + A31 * (epsilon ** 3) * np.sinh(k_z_plus_h) * self.sea_state.omega_det[s]
                                * -np.cos(psi)
                                + A33 * (epsilon ** 3) * np.sinh(3 * k_z_plus_h) * 3 * self.sea_state.omega_det[s]
                                * -np.cos(3 * psi)
                                + A42 * (epsilon ** 4) * np.sinh(2 * k_z_plus_h) * 2 * self.sea_state.omega_det[s]
                                * -np.cos(2 * psi)
                                + A44 * (epsilon ** 4) * np.sinh(4 * k_z_plus_h) * 4 * self.sea_state.omega_det[s]
                                * -np.cos(4 * psi)
                                + A51 * (epsilon ** 5) * np.sinh(k_z_plus_h) * self.sea_state.omega_det[s]
                                * - np.cos(psi)
                                + A53 * (epsilon ** 5) * np.sinh(3 * k_z_plus_h) * 3 * self.sea_state.omega_det[s]
                                * -np.cos(3 * psi)
                                + A55 * (epsilon ** 5) * np.sinh(5 * k_z_plus_h) * 5 * self.sea_state.omega_det[s]
                                * -np.cos(5 * psi))

                    if z > self.eta[i_t, s]:
                        self.u[i_t, i_z, s] = self.w[i_t, i_z, s] = self.du[i_t, i_z, s] = self.dw[i_t, i_z, s] = 0
        return self
