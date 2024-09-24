import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from dataclasses import dataclass
import os

@dataclass 
class FrqDrcSpectrum():

    omega: np.ndarray
    phi: np.ndarray
    omega_p: float
    phi_m: float
    alpha: float
    
    gamma = 3.3  # make larger to decrease width of Jonswap
    r = 5.
    beta = 4.
    nu = 2.7
    sig_l = 0.05  # make smaller to decrease directional spreading
    sig_r = 0 # 0.26  # make zero to decrease directional spreading

    def compute_spectrum(self) -> np.ndarray:
        """returns frequency direction spectrum for a single angular frequency and direction.

        Returns:
            dens (np.ndarray): freq direction spectrum 
        """

        S = np.empty((len(self.omega), len(self.phi)))

        for i_om, om in enumerate(self.omega):
            for i_phi, phi in enumerate(self.phi):
                S[i_om, i_phi] = frq_dr_spctrm(om, phi, self.alpha, self.omega_p, self.gamma, self.r, self.phi_m, self.beta, self.nu, self.sig_l, self.sig_r)
        
        return S

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
    phi_m: float
 
    @property
    def fp(self) -> float:
        """get peak frequency

        Returns:
            float: peak frequency
        """
        return 1/self.tp

    @property
    def omega_p(self) -> float:
        """get peak angular frequency

        Returns:
            float: peak angular frequency
        """
        return 2 * np.pi / self.tp

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

    @property
    def phi_values(self) -> np.ndarray:
        """returns the phi_values to evaluate at

        Returns:
            np.ndarray: phi_values
        """
        return np.linspace(0, 2*np.pi, 32)

    @property
    def nphi(self) -> int:
        """returns number of phi values

        Returns:
            int: number of phi values to evaluate at
        """
        return len(self.phi_values)

    @property
    def dphi(self) -> float:
        """returns steps in phi-values

        Returns:
            float: phi step size (homogenous)
        """
        return self.phi_values[1] - self.phi_values[0]

    @property
    def frequency(self) -> np.ndarray:
        """calculates and returns the frequencies to evaluate the spectrum at for given period and sample f

        Returns:
            np.ndarray: contributing frequencies
        """

        f_range = np.linspace(1e-3, self.nt - 1, int(self.nt)) / (self.nt / self.sample_f)  # selecting frequency

        return f_range

    @property
    def omega_values(self) -> np.ndarray:
        """calculates and returns the angular frequencies to evaluate the spectrum at for given period and sample f

        Returns:
            np.ndarray: contributing angular frequencies
        """

        omega_values = 2 * np.pi * self.frequency

        return omega_values
    
    @property
    def domega(self) -> float:
        """returns the angular frequency step size

        Returns:
            float: angular frequency step size
        """
        return self.omega_values[1] - self.omega_values[0]

    @property
    def spectrum(self) -> FrqDrcSpectrum:
        """returns the frequency direction spectrum

        Returns:
            FrqDrcSpectrum: frequency direction spectrum
        """
        unnorm_spect = FrqDrcSpectrum(omega=self.omega_values, phi=self.phi_values, omega_p=self.omega_p, phi_m=self.phi_m, alpha=1.0)
        m0 = np.sum(unnorm_spect.compute_spectrum() * self.domega * self.dphi)
    
        m0_target = self.hs**2 / 16
        alpha = m0_target / m0

        return FrqDrcSpectrum(omega=self.omega_values, phi=self.phi_values, omega_p=self.omega_p, phi_m=self.phi_m, alpha=alpha)

    def compute_elevation(self, cond:bool, cond_crest:float) -> np.ndarray:
        """
        returns random wave surface with frequency direction spectrum defined below

        """

        # get directional spectrum values - will vectorise this later
        S = self.spectrum.compute_spectrum() * self.domega * self.dphi

        # get random coefficients
        A = np.random.randn(self.nt, self.nphi) * S
        B = np.random.randn(self.nt, self.nphi) * S


            # if cond:
            #     m = 0

            #     c = self.spctr[s].df * self.spctr[s].density
            #     d = self.spctr[s].df * self.spctr[s].density * self.spctr[s].omega

            #     Q = (a[s] - np.sum(A))/np.sum(c)
            #     R = (m - np.sum(self.spctr[s].omega * B))/np.sum(d*self.spctr[s].omega)

            #     A = A + Q * c
            #     B = B + R * d


# %% Conditional Wave simulation

#             if CndFlg

#                 %Adjust A& B to

#                 sum_A = sum(sum(A,1),2); %sum over frequencies and theta

#                 % \sum_{i} A_{i}*\omega_{i} (over frequency)

#                 sum_Bf = sum(sum(B .* Spec.f, 1), 2);

#                 Spec = Spec.SpectralMoment([0,2]);

#                 % \lambda^{2}

#                 T_2sq = Spec.Moment.M0 ./ Spec.Moment.M2; % sample T2

#                 A=A + (C_0 - sum_A) .* Spec.S ./ Spec.Moment.M0;

#                 B=B - (T_2sq .* sum_Bf) .* (Spec.S .* fSim) ./ Spec.Moment.M0;

#             end

        if cond:

            # Tp: 10
        #    T1: 8.3885
        #    T2: 7.9211

            m0 = np.sum(S)
            m2 = np.sum(S * (self.omega_values.reshape(self.nt, 1)**2)) / (2*np.pi)**2
            t2sqr = m0 / m2
            t2 = np.sqrt(t2sqr)

            sum_A = np.sum(A)
            sum_Bomega = np.sum(B * self.omega_values.reshape(self.nt, 1))

            A = A + (cond_crest - sum_A) * S / m0
            B = B - (t2sqr * sum_Bomega) * (S * self.omega_values.reshape(self.nt, 1)) / m0

        i = complex(0, 1)
        Z = A + i*B

        # get wave numbers
        k = alt_solve_dispersion(self.omega_values, self.depth)
        kx = np.outer(np.cos(2*np.pi*self.phi_values), k)
        ky = np.outer(np.sin(2*np.pi*self.phi_values), k)

        # do fft
        k_vec = np.einsum('i,jk->ijk', self.x_values, kx) + np.einsum('i,jk->ijk', self.y_values, ky)
        Z_vec = np.sum(np.exp(i * k_vec) * np.transpose(Z).reshape(1, self.nphi, self.nt), 1)
        eta = np.fft.fftshift(np.real(np.fft.fft(Z_vec, self.nt, 1)), 1)

        return eta


def frq_dr_spctrm(omega: np.ndarray, phi: np.ndarray, alpha: float, om_p: float, gamma: float,
                  r: float, phi_m: float, beta: float, nu: float, sig_l: float,
                  sig_r: float):
    """returns frequency direction spectrum for a single angular frequency and direction.

    Args:
        omega (np.ndarray): angular frequency
        phi (np.ndarray): direction (from)
        alpha (float): scaling parameter
        om_p (float): peak ang freq
        gamma (float): peak enhancement factor
        r (float): spectral tail decay index
        phi_m (float): mean direction
        beta (float): limiting peak separation
        nu (float): peak separation shape
        sig_l (float): limiting angular width
        sig_r (float): angular width shape

    Returns:
        dens (np.ndarray): freq direction spectrum 
    """
    dens = sprd_fnc(omega, phi, om_p, phi_m, beta, nu, sig_l, sig_r) * d_jonswap(omega, alpha, om_p, gamma, r)

    return dens

def alt_solve_dispersion(omega: np.ndarray, d: float) -> float:
    """uses method of (Guo, 2002) to solve dispersion relation for k

    Args:
        omega (np.ndarray): angular frequency [s^-1]
        d (float): water depth [m]

    Returns:
        k (float): wave number [m^-1]
    """

    g = 9.81
    beta = 2.4901

    x = d * omega / np.sqrt(g * d)

    y = x**2 * (1 - np.exp(-x**beta))**(-1/beta)

    k = y / d

    return k

def sprd_fnc(omega: float, phi: float, om_p: float, phi_m: float, beta: float, nu: float,
             sig_l: float, sig_r: float):
    """returns bimodal wrapped Gaussian spreading function D(omega, phi) at a single point

    Args:
        omega (float): angular frequency
        phi (float): direction (from)
        om_p (float): peak ang freq
        phi_m (float): mean direction
        beta (float): limiting peak separation
        nu (float): peak separation shape
        sig_l (float): limiting angular width
        sig_r (float): angular width shape

    Returns:
        dens (float): D(omega, phi) for given omega and phi
    """
    k_num = 200
    k_range = np.linspace(start=-k_num/2, stop=k_num/2, num=k_num + 1)

    phi_m1 = phi_m + beta * np.exp(-nu * min(om_p / np.abs(omega), 1)) / 2
    phi_m2 = phi_m - beta * np.exp(-nu * min(om_p / np.abs(omega), 1)) / 2
    phi_arr = np.array([phi_m1, phi_m2])

    sigma = sig_l - sig_r / 3 * (4 * (om_p / np.abs(omega)) ** 2 - (om_p / np.abs(omega)) ** 8)

    nrm_cnst = (2 * sigma * np.sqrt(2 * np.pi)) ** -1
    dens_k = np.empty(k_num + 1)

    for i_k, k in enumerate(k_range):
        exp_term = np.exp(-0.5 * ((phi - phi_arr - 2 * np.pi * k) / sigma) ** 2)
        dens_k[i_k] = np.sum(exp_term)

    dens = nrm_cnst * np.sum(dens_k)

    return dens


def d_jonswap(omega: float, alpha: float, om_p: float, gamma: float, r: float):
    """jonswap density using formulation used in Jake's paper

    Args:
        omega (float): angular frequency
        alpha (float): scaling parameter
        om_p (float): peak ang freq
        gamma (float): peak enhancement factor
        r (float): spectral tail decay index

    Returns:
        dens (float): JONSWAP density for given omega
    """

    delta = np.exp(-(2 * (0.07 + 0.02 * (om_p > np.abs(omega)))) ** -2 * (np.abs(omega) / om_p - 1) ** 2)

    dens = alpha * omega ** -r * np.exp(-r / 4 * (np.abs(omega) / om_p) ** -4) * gamma ** delta

    return dens


if __name__ == "__main__":

    np.random.seed(0)

    # define parameters
    sample_f = 4.00
    period = 60
    x_range = np.linspace(-100, 100, 40)
    y_range = np.linspace(-100, 100, 40)
    z_range = np.linspace(-100, 100, 20)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    hs = 12.4
    tp = 10
    phi_m = np.pi

    # create instance
    spatial_wave = SpatialLinearKin(sample_f=sample_f, period=period, x_values=x_grid.flatten(), y_values=y_grid.flatten(), z_values=z_range, hs=hs, tp=tp, phi_m=phi_m)

    # get elevation
    c = 100
    eta = spatial_wave.compute_elevation(cond=True, cond_crest=c)

    # make gif of 3d wave evolution over time usign 3d plot
    path = '/home/speersm/GitHub/force_calculation_and_wave_sim/test_scripts/spatial_wave_sim/'
    os.system(f'mkdir -p {path}temp/')
    for i in range(spatial_wave.nt):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X=x_grid, Y=y_grid, Z=eta[:,i].reshape(len(x_range),len(y_range)))
        ax.set_zlim(-7, c+5)
        plt.title(f"Time: {i}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f"{path}temp/wave_{i:03}.png")
        plt.close()

    # use pngs to make gif
    os.system(f'convert -delay 20 -loop 0 {path}temp/wave_*.png {path}wave.gif')

    # delete pngs
    os.system(f'rm {path}temp/*.png')
    os.system(f'rmdir {path}temp/')

   