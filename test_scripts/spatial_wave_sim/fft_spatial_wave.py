import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from dataclasses import dataclass

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
class FrqDrcSpectrum():

    omega: np.ndarray
    phi: np.ndarray
    omega_p: float
    phi_m: float

    alpha = 0.7
    gamma = 3.3  # make larger to decrease width of Jonswap
    r = 5.
    beta = 4.
    nu = 2.7
    sig_l = 0.55  # make smaller to decrease directional spreading
    sig_r = 0.26  # make zero to decrease directional spreading

    def compute_spectrum(self) -> np.ndarray:
        """returns frequency direction spectrum for a single angular frequency and direction.

        Returns:
            dens (np.ndarray): freq direction spectrum [] (??, ??)
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
        return FrqDrcSpectrum(omega=self.omega_values, phi=self.phi_values, omega_p=self.omega_p, phi_m=self.phi_m)


    def compute_elevation(self, cond:bool) -> np.ndarray:
        """
        returns random wave surface with frequency direction spectrum defined below

        """

        # get directional spectrum values - will vectorise this later
        S = self.spectrum.compute_spectrum()

        A = np.random.randn(self.nt, self.nphi) * S * self.domega * self.dphi
        B = np.random.randn(self.nt, self.nphi) * S * self.domega * self.dphi

        return A, B


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
        dens (np.ndarray): freq direction spectrum [] (??, ??)
    """
    dens = sprd_fnc(omega, phi, om_p, phi_m, beta, nu, sig_l, sig_r) * d_jonswap(omega, alpha, om_p, gamma, r)

    return dens


def solve_dispersion(omega: float, h: float, upp: float):
    """returns wave number k for given angular frequency omega
    Args:
        omega (float): angular frequency [s^-1]
        h (float): water depth [metres]
        upp (float): upper limit of interval to find k over []

    Returns:
        k (float): wave number [m^-1]
    """

    k = optimize.bisect(f=dispersion_diff, a=1e-7, b=upp, args=(h, omega))

    return k


def dispersion_diff(k: float, h: float, omega: float):
    """function to optimise in solve_dispersion
    Args:
        k (float): wave number
        h (float): water depth
        omega (float): angular frequency

    Returns:
        diff (float): difference to find zero in solve_dispersion
    """
    g = 9.81

    diff = omega ** 2 - g * k * np.tanh(k * h)

    return diff


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

    # define parameters
    sample_f = 1.00
    period = 60
    x_range = np.linspace(-100, 100, 100)
    y_range = np.linspace(-100, 100, 100)
    z_range = np.linspace(-100, 100, 100)
    tp = 10
    phi_m = 0

    # create instance
    spatial_wave = SpatialLinearKin(sample_f=sample_f, period=period, x_values=x_range, y_values=y_range, z_values=z_range, tp=tp, phi_m=phi_m)

    # compute elevation
    A, B = spatial_wave.compute_elevation(cond=False)

    # print(A, B)
    print("done")