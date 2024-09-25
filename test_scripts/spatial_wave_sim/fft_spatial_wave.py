import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from dataclasses import dataclass
import os
import warnings
import math

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

    def compute_kinematics(self, cond:bool, cond_crest:float) -> np.ndarray:
        """
        returns random wave surface and kinematics with frequency direction spectrum defined below

        """

        # get directional spectrum values - will vectorise this later
        S = self.spectrum.compute_spectrum() * self.domega * self.dphi

        # get random coefficients
        A = np.random.randn(self.nt, self.nphi) * S
        B = np.random.randn(self.nt, self.nphi) * S

        # if conditioning on a crest height at 0, 0, 0
        if cond:

            c = S
            d = S * self.omega_values.reshape(self.nt, 1)

            Q = (cond_crest - np.sum(A))/np.sum(c)
            R = (0 - np.sum(self.omega_values.reshape(self.nt, 1) * B))/np.sum(d*self.omega_values.reshape(self.nt, 1))

            A = A + Q * c
            B = B + R * d


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

        # compute kinematics
        d = self.depth
        u = np.empty((self.nx, self.nt, self.nz))
        w = np.empty((self.nx, self.nt, self.nz))
        du = np.empty((self.nx, self.nt, self.nz))
        dw = np.empty((self.nx, self.nt, self.nz))
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

            Z2 = A + B*i
            Z_vec2 = np.sum(np.exp(i * k_vec) * np.transpose(Z2).reshape(1, self.nphi, self.nt) 
            * self.omega_values.reshape(1, 1, self.nt) * qf1.reshape(1, 1, self.nt), 1)
            u[:, :, i_z] = np.fft.fftshift(np.real(np.fft.fft(Z_vec2, self.nt, 1)), 1) * (z_init < eta)

            Z3 = B - A*i
            Z_vec3 = np.sum(np.exp(i * k_vec) * np.transpose(Z3).reshape(1, self.nphi, self.nt)
            * (self.omega_values.reshape(1, 1, self.nt)**2) * qf1.reshape(1, 1, self.nt), 1)
            du[:, :, i_z] = np.fft.fftshift(np.real(np.fft.fft(Z_vec3, self.nt, 1)), 1) * (z_init < eta)

            Z4 = B - A*i
            Z_vec4 = np.sum(np.exp(i * k_vec) * np.transpose(Z4).reshape(1, self.nphi, self.nt)
            * self.omega_values.reshape(1, 1, self.nt) * qf2.reshape(1, 1, self.nt), 1)
            w[:, :, i_z] = np.fft.fftshift(np.real(np.fft.fft(Z_vec4, self.nt, 1)), 1) * (z_init < eta)

            Z5 = -A - B*i
            Z_vec5 = np.sum(np.exp(i * k_vec) * np.transpose(Z5).reshape(1, self.nphi, self.nt)
            * (self.omega_values.reshape(1, 1, self.nt)**2) * qf2.reshape(1, 1, self.nt), 1)
            dw[:, :, i_z] = np.fft.fftshift(np.real(np.fft.fft(Z_vec5, self.nt, 1)), 1) * (z_init < eta)

            # g3 = (B-A*i) * (2*np.pi*self.spctr[s].frequency)**2 * qf1
            # g4 = (B-A*i) * (2*np.pi*self.spctr[s].frequency) * qf2
            # g5 = (-A-B*i) * (2*np.pi*self.spctr[s].frequency)**2 * qf2

            # self.u[:, i_z, s] = np.real(fftshift(fft(g2))) * (z_init < self.eta[:, s]) + np.cos(self.sea_state.current_incidence) * self.sea_state.current
            # self.du[:, i_z, s] = np.real(fftshift(fft(g3))) * (z_init < self.eta[:, s])
            # self.w[:, i_z, s] = np.real(fftshift(fft(g4))) * (z_init < self.eta[:, s])
            # self.dw[:, i_z, s] = np.real(fftshift(fft(g5))) * (z_init < self.eta[:, s])

        return eta, u, du, w, dw


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
    z_range = np.linspace(-100, 100, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    hs = 12.4
    tp = 10
    phi_m = np.pi

    # create instance
    spatial_wave = SpatialLinearKin(sample_f=sample_f, period=period, x_values=x_grid.flatten(), y_values=y_grid.flatten(), z_values=z_range, hs=hs, tp=tp, phi_m=phi_m)

    # get elevation
    c = 50
    eta, u, du, w, dw = spatial_wave.compute_kinematics(cond=True, cond_crest=c)

    # plot a slice of the wave kinematics at time 0
    # create grid
    zt_grid = np.meshgrid(z_range, spatial_wave.t_values)

    # plot
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(spatial_wave.t_values, eta[779,:])
    plt.scatter(zt_grid[1].flatten(), zt_grid[0].flatten(), s=1, c=u[779, :, :].flatten())
    plt.title("u")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.plot(spatial_wave.t_values, eta[779,:])
    plt.scatter(zt_grid[1].flatten(), zt_grid[0].flatten(), s=1, c=du[779, :, :].flatten())
    plt.title("du")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.plot(spatial_wave.t_values, eta[779,:])
    plt.scatter(zt_grid[1].flatten(), zt_grid[0].flatten(), s=1, c=w[779, :, :].flatten())
    plt.title("w")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.plot(spatial_wave.t_values, eta[779,:])
    plt.scatter(zt_grid[1].flatten(), zt_grid[0].flatten(), s=1, c=dw[779, :, :].flatten())
    plt.title("dw")
    plt.xlabel("t")
    plt.ylabel("z")
    plt.colorbar()
    plt.show()

    # save figure
    path = '/home/speersm/GitHub/force_calculation_and_wave_sim/test_scripts/spatial_wave_sim/'
    save_path = path + 'wave_kinematics.png'
    plt.savefig(save_path)
    
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

   