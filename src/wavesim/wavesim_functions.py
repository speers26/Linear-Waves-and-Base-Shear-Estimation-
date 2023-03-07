import numpy as np
from scipy import optimize
from scipy.fft import fft, fftshift

# need to fix the docstrings for a few of these


# for single location random waves

def ptws_random_wave_sim(t: float, z: float, depth: float, a: float, om_range: np.ndarray, spctrl_dens: np.ndarray, cond: bool):
    """returns pointwave surface level eta and kinematics for x=0

    Args:
        t (float): time [s]
        z (float): height in water [m]
        d (float): water depth [m]
        a (float): wave height at t=0 [m]
        om_range (np.ndarray): range of contributing angular frequencies [s^-1]
        spctrl_dens (np.ndarray): spectrum corresponding to om_range
        cond (bool): True if we want a conditional wave simulation

    Returns:
        eta (float): surface level [m]
        u_x (float): horizontal velocity [ms^-1]
        u_z (float): vertical velocity [ms^-1]
        du_x (float): horizontal acceleration [ms^-2]
        du_z (float) vertical acceleration [ms^-2]
    """

    np.random.seed(1234)

    f_num = len(om_range)
    df = (om_range[1] - om_range[0]) / (2*np.pi)

    A = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)
    B = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)

    if cond:
        m = 0

        c = df * spctrl_dens
        d = df * spctrl_dens * om_range

        Q = (a - np.sum(A))/np.sum(c)
        R = (m - np.sum(om_range * A))/np.sum(d*om_range)

        A = A + Q * c
        B = B + R * d

    eta = np.sum(A * np.cos(om_range*t) + B * np.sin(om_range*t))

    d = depth

    z_init = z
    z = d * (d + z) / (d + eta) - d   # for Wheeler stretching

    k = np.empty(f_num)
    for i_om, om in enumerate(om_range):
        # k[i_om] = solve_dispersion(omega=om, h=d, upp=75)
        k[i_om] = alt_solve_dispersion(omega=om, d=d)

    u_x = np.sum((A * np.cos(om_range*t) + B * np.sin(om_range*t)) * om_range * (np.cosh(k*(z+d))) / (np.sinh(k*d)))
    u_z = np.sum((-A * np.sin(om_range*t) + B * np.cos(om_range*t)) * om_range * (np.sinh(k*(z+d))) / (np.sinh(k*d)))

    du_x = np.sum((-A * np.sin(om_range*t) + B * np.cos(om_range*t)) * om_range**2 * (np.cosh(k*(z+d)))
                  / (np.sinh(k*d)))
    du_z = np.sum((-A * np.cos(om_range*t) - B * np.sin(om_range*t)) * om_range**2 * (np.sinh(k*(z+d)))
                  / (np.sinh(k*d)))

    if z_init > eta:
        u_x = u_z = du_x = du_z = 0

    return eta, u_x, u_z, du_x, du_z


def fft_random_wave_sim(z_range: np.ndarray, d: np.ndarray, a: float, om_range: np.ndarray, spctrl_dens: np.ndarray, cond: bool):
    """generates random wave surface and kinematics using FFT

    Args:
        z_range (np.ndarray): range of depths [m]
        d (float): water depth
        a (float): wave height at t=0 [m]
        om_range (np.ndarray): range of angular velocities [s^-1]
        spctrl_dens (np.ndarray): spectrum corresponding to om_range
        cond (bool): True if we want a conditional wave simulation

    Returns:
        eta (np.ndarray): wave surface height [m]
        u_x (np.ndarray): horizontal velociy at given z [ms^-1]
        u_v (np.ndarray): vertical velocity at given z [ms^-1]
        du_x (np.ndarray): horizontal acceleration at given z [ms^-2]
        du_v (np.ndarray): vertical acceleration at given z [ms^-2]
    """

    water_depth = d
    np.random.seed(1234)

    f_range = om_range / (2*np.pi)
    f_num = len(f_range)
    df = f_range[1] - f_range[0]

    A = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)
    B = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)

    if cond:
        m = 0

        c = df * spctrl_dens
        d = df * spctrl_dens * om_range

        Q = (a - np.sum(A))/np.sum(c)
        R = (m - np.sum(om_range * B))/np.sum(d*om_range)

        A = A + Q * c
        B = B + R * d

    i = complex(0, 1)
    g1 = A + B * i

    eta = np.real(np.fft.fftshift(np.fft.fft(g1)))

    k = np.empty(f_num)

    d = water_depth

    for i_f, f in enumerate(f_range):
        omega = 2 * np.pi * f
        # k[i_f] = rws.solve_dispersion(omega, d, 95)
        k[i_f] = alt_solve_dispersion(omega, d)

    u_x = np.empty((f_num, len(z_range)))
    du_x = np.empty((f_num, len(z_range)))
    u_z = np.empty((f_num, len(z_range)))
    du_z = np.empty((f_num, len(z_range)))

    for i_z, z in enumerate(z_range):

        z_init = z
        if z > -3:
            z = -3

        g2 = (A+B*i) * 2*np.pi*f_range * (np.cosh(k*(z + d))) / (np.sinh(k*d))
        g3 = (B-A*i) * (2*np.pi*f_range)**2 * (np.cosh(k*(z+d))) / (np.sinh(k*d))
        g4 = (B-A*i) * (2*np.pi*f_range) * (np.sinh(k*(z+d))) / (np.sinh(k*d))
        g5 = (-A-B*i) * (2*np.pi*f_range)**2 * (np.sinh(k*(z+d))) / (np.sinh(k*d))

        u_x[:, i_z] = np.real(np.fft.fftshift(np.fft.fft(g2))) * (z_init < eta)
        du_x[:, i_z] = np.real(np.fft.fftshift(np.fft.fft(g3))) * (z_init < eta)
        u_z[:, i_z] = np.real(np.fft.fftshift(np.fft.fft(g4))) * (z_init < eta)
        du_z[:, i_z] = np.real(np.fft.fftshift(np.fft.fft(g5))) * (z_init < eta)

    return eta, u_x, u_z, du_x, du_z


def alt_solve_dispersion(omega: float, d: float):
    """uses method of (Guo, 2002) to solve dispersion relation for k

    Args:
        omega (float): angular frequency [s^-1]
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


def djonswap(f: np.ndarray, hs: float, tp: float):

    """
    returns JONSWAP density for given frequency range

    Args:
        f (np.ndarray): frequency [s^-1]
        hs (float): significant wave height [m]
        tp (float): significant wave period [s]

    Returns:
        dens (np.ndarray): JONSWAP density for given frequencies []
    """
    g = 9.81
    fp = 1. / tp
    df = f[1] - f[0]

    # example constants from https://wikiwaves.org/Ocean-Wave_Spectra
    gamma = 2  # 3.3
    sigma_a = 0.07
    sigma_b = 0.09

    sigma = (f < fp) * sigma_a + (f >= fp) * sigma_b

    gamma_coeff = gamma ** np.exp(-0.5 * (((f / fp - 1)/sigma) ** 2))
    dens = g ** 2 * (2 * np.pi) ** -4 * f ** -5 * np.exp(-1.25 * (tp*f) ** -4) * gamma_coeff

    area = sum(dens*df)

    dens *= hs ** 2 / (16 * area)

    return dens


def random_waves_acf(tau: np.ndarray, f: np.ndarray, spctrl_dens: np.ndarray):
    """find acf function of the gaussian random wave surface with given spectrum

    Args:
        tau (np.ndarray): lags []
        f (np.ndarray): contributing frequencies [hertz]
        spctrl_dens (np.ndarray): spectral densities for given frequencies []

    Returns:
        acf (np.ndarray): auto correlation
    """

    df = f[1] - f[0]
    spctrl_area = np.sum(spctrl_dens * df)

    outer_ft = np.outer(f, tau)    # (n_freq x tau_length)

    acf_mat = np.cos(2 * np.pi * outer_ft) * spctrl_dens[:, np.newaxis] * df / spctrl_area    # (n_freq x tau_length)
    acf_vec = np.sum(acf_mat, axis=0)   # sum over columns to give (1 x tau_length)

    return acf_vec


def kth_moment(k: int, f: np.ndarray, spctrl_dens: np.ndarray):
    """function to return the kth moment of the given spectrum evaulated at given frequencies

    Args:
        k (int): moment
        f_seq (np.ndarray): frequencies [hertz]

    Returns:
        k_integral (_type_): integral equal to the kth moment
    """

    df = f[1] - f[0]

    k_integral = np.sum(spctrl_dens * f ** k * df)

    return k_integral


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


def morison_load(u:np.ndarray, du: np.ndarray, diameter=1.0, rho=1024.0, c_m=1.0, c_d=1.0):
    """compute unit Morison load for a vertical cylinder

    Args:
        u (np.ndarray): horizontal velocity [m/s]
        du (np.ndarray): horizontal acceleration [m/s^2]
        diameter (float, optional): diameter of cylinder. Defaults to 1.0. [m]
        rho (float, optional): water density. Defaults to 1024.0. [kg/m^3]
        c_m (float, optional): a coefficient. Defaults to 1.0. [unitless]
        c_d (float, optional): a coefficient. Defaults to 1.0. [unitless]

    Returns:
        np.ndarray: horizontal unit morrison load [N/m]
    """

    return rho * c_m * (np.pi / 4) * (diameter ** 2) * du + 0.5 * rho * c_d * diameter * u * np.abs(u)


# for spatial random wave surface

def spatial_random_wave(om_range: np.ndarray, phi_range: np.ndarray, Dr_spctrm: np.ndarray, t: np.ndarray, x_range: np.ndarray,
                                       y_range: np.ndarray, h: float):
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
    dens = sprd_fnc(omega, phi, om_p, phi_m, beta, nu, sig_l, sig_r) * alt_djonswap(omega, alpha, om_p, gamma, r)

    return dens


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


def alt_djonswap(omega: float, alpha: float, om_p: float, gamma: float, r: float):
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


# for crest distributions

def rayleigh_cdf(eta: np.ndarray, hs: float):
    """returns the rayleigh cdf

    Args:
        eta (np.ndarray): crest heights
        hs (float): sig wave height

    Returns:
        p (np.ndarray): rayleigh probability
    """

    p = 1 - np.exp(-8 * eta**2 / hs**2)

    return p


def rayleigh_pdf(eta: np.ndarray, hs: float):
    """_summary_

    Args:
        eta (np.ndarray): _description_
        hs (float): _description_
    """

    d = -np.exp(-8 * eta**2 / hs**2) * -8 * 2 * eta / hs**2

    return d
