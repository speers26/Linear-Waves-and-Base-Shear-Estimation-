'''
Code for generating kinematics, wave velocity, wave accelerations

'''
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.fft import fft, fftshift
from wavesim.dispersion import alt_solve_dispersion, solve_dispersion
from wavesim.spectrum import Spectrum

def airy_kinematics(k: np.ndarray, h: np.ndarray, A: np.ndarray, x: np.ndarray,
                    omega: np.ndarray, t: np.ndarray, z: np.ndarray):
    """Generates Airy kinematics

    Args:
        k (np.ndarray): wave number
        h (np.ndarray): water depth
        A (np.ndarray): amplitude
        x (np.ndarray): spatial position
        omega (np.ndarray): angular frequency
        t (np.ndarray): temporal position
        z (np.ndarray): depth
    """
    g = 9.81

    eta = A * np.sin(omega * t - k * x)

    u = omega * A * ((np.cosh(k * (h + z))) / (np.sinh(k * h))) * np.sin(omega * t - k * x)

    w = omega * A * ((np.sinh(k * (h + z))) / (np.sinh(k * h))) * np.cos(omega * t - k * x)

    du = omega ** 2 * A * ((np.cosh(k * (h + z))) / (np.sinh(k * h))) * np.cos(omega * t - k * x)

    dw = -omega ** 2 * A * ((np.sinh(k * (h + z))) / (np.sinh(k * h))) * np.sin(omega * t - k * x)

    if z > eta:
        u = w = du = dw = 0

    return eta, u, w, du, dw


def stokes_kinematics(k: np.ndarray, h: np.ndarray, A: np.ndarray, x: np.ndarray,
                      omega: np.ndarray, t: np.ndarray, theta: np.array, z: np.ndarray):
    """ Generate Stokes kinematic profiles

    Args:
        k (np.ndarray): wave number (coming from Stokes disp.)
        h (np.ndarray): wave depth TODO: rename to d
        A (np.ndarray): wave amplitude: A = H/2 -> you will get an asymmetric profile
        x (np.ndarray): spatial position [m]
        omega (np.ndarray): angular frequency [rad/s]
        t (np.ndarray): time [s]
        theta (np.array): wave direction??????? TODO: check
        z (np.ndarray): depth (relative to mean sea level?????????) TODO: check

    Returns:
        eta  = surface elevation [m]
        u = horizontal velocity [m/s]
        w = vertical velocity [m/s]
        dudt = horizontal acceleration [m/s**2]
        dwdt = vertical acceleration [m/(s**2)]
    """
    g = 9.81

    kd = k * h
    # !e Initialisation
    S = 1 / np.cosh(2 * kd)
    # Calculation of the A coefficients
    Aco = np.empty(9)
    Aco[0] = 1 / np.sinh(kd)
    Aco[1] = 3 * (S ** 2) / (2 * ((1. - S) ** 2))
    Aco[2] = (-4 - 20 * S + 10 * (S ** 2) - 13 * (S ** 3)) / (8 * np.sinh(kd) * ((1 - S) ** 3))
    Aco[3] = (-2 * (S ** 2) + 11 * (S ** 3)) / (8 * np.sinh(kd) * ((1.-S) ** 3))
    Aco[4] = (12 * S - 14 * (S ** 2) - 264 * (S ** 3) - 45 * (S ** 4) - 13 * (S ** 5)) / (24*((1.-S)**5))
    Aco[5] = (10 * (S ** 3) - 174 * (S ** 4) + 291 * (S ** 5) + 278 * (S ** 6)) / (48 * (3 + 2 * S) * ((1 - S) ** 5))
    Aco[6] = (-1184 + 32 * S + 13232 * (S ** 2) + 21712 * (S ** 3) + 20940 * (S ** 4) + 12554 * (S ** 5) - 500 *
              (S ** 6) - 3341 * (S ** 7) - 670 * (S ** 8)) / (64 * np.sinh(kd) * (3 + 2 * S) * (4 + S) * ((1 - S) ** 6))
    Aco[7] = (4 * S + 105 * (S ** 2) + 198 * (S ** 3) - 1376 * (S ** 4) - 1302 * (S ** 5) - 117 * (S ** 6) +
              58 * (S ** 7))/(32 * np.sinh(kd) * (3 + 2 * S) * ((1 - S) ** 6))
    Aco[8] = (-6 * (S ** 3) + 272 * (S ** 4) - 1552 * (S ** 5) + 852 * (S ** 6) + 2029 * (S ** 7) + 430 * (S ** 8)) \
        / (64 * np.sinh(kd) * (3 + 2 * S) * (4 + S) * ((1 - S) ** 6))
    # Calculation of the B coefficients
    Bco = np.empty(6)
    Bco[0] = (1 / np.tanh(kd)) * (1 + 2 * S) / (2 * (1 - S))
    Bco[1] = -3 * (1 + 3 * S + 3 * (S ** 2) + 2 * (S ** 3)) / (8 * ((1 - S) ** 3))
    Bco[2] = (1 / np.tanh(kd)) * (6 - 26 * S - 182 * (S ** 2) - 204 * (S ** 3) - 25 * (S ** 4) + 26 * (S ** 5)) \
        / (6 * (3 + 2 * S) * ((1 - S) ** 4))
    Bco[3] = (1./np.tanh(kd)) * (24 + 92 * S + 122 * (S ** 2) + 66 * (S ** 3) + 67 * (S ** 4) + 34 * (S ** 5)) \
        / (24 * (3 + 2 * S) * ((1 - S) ** 4))
    Bco[4] = 9 * (132 + 17 * S - 2216 * (S ** 2) - 5897 * (S ** 3) - 6292 * (S ** 4) - 2687 * (S ** 5)
                  + 194 * (S ** 6) + 467 * (S ** 7) + 82 * (S ** 8)) / (128 * (3 + 2 * S) * (4 + S) * ((1 - S) ** 6))
    Bco[5] = 5 * (300 + 1579 * S + 3176 * (S ** 2) + 2949 * (S ** 3) + 1188 * (S ** 4) + 675 * (S ** 5)
                  + 1326 * (S ** 6) + 827 * (S ** 7) + 130 * (S ** 8)) / (384 * (3 + 2 * S) * (4 + S) * ((1 - S) ** 6))
    # Calculation of the C coefficients
    Cco = np.empty(3)
    Cco[0] = np.sqrt(np.tanh(kd))
    Cco[1] = (np.sqrt(np.tanh(kd)) * (2 + 7 * S ** 2)) / (4 * (1-S) ** 2)
    Cco[2] = (np.sqrt(np.tanh(kd)) * (4 + 32 * S - 116 * S ** 2 - 400 * S ** 3 - 71 * S ** 4 + 146 * S ** 5)) \
        / (32 * (1 - S) ** 5)
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
    epsilon = A * k
    #
    psi = k * x - omega * t

    k_z_plus_h = k * (z + h)
    # z
    eta = (1 / k) * (epsilon * np.cos(psi) + B22 * (epsilon ** 2) * np.cos(2 * psi)
                     + B31 * (epsilon ** 3) * (np.cos(psi) - np.cos(3 * psi))
                     + (epsilon ** 4) * (B42 * np.cos(2 * psi) + B44 * np.cos(4 * psi))
                     + (epsilon ** 5) * (-(B53 + B55) * np.cos(psi) + B53 * np.cos(3 * psi)
                     + B55 * np.cos(5 * psi)))
    # u calculation
    u = (C0 * np.sqrt(g / k ** 3)) * (k * np.cos(theta)) \
        * (A11 * epsilon * np.cosh(k_z_plus_h) * np.cos(psi)
           + A22 * (epsilon ** 2) * np.cosh(2 * k_z_plus_h) * 2 * np.cos(2 * psi)
           + A31 * (epsilon ** 3) * np.cosh(k_z_plus_h) * np.cos(psi)
           + A33 * (epsilon ** 3) * np.cosh(3 * k_z_plus_h) * 3 * np.cos(3 * psi)
           + A42 * (epsilon ** 4) * np.cosh(2 * k_z_plus_h) * 2 * np.cos(2 * psi)
           + A44 * (epsilon ** 4) * np.cosh(4 * k_z_plus_h) * 4 * np.cos(4 * psi)
           + A51 * (epsilon ** 5) * np.cosh(k_z_plus_h) * np.cos(psi) \
           + A53 * (epsilon ** 5) * np.cosh(3 * k_z_plus_h) * 3 * np.cos(3 * psi) \
           + A55 * (epsilon ** 5) * np.cosh(5 * k_z_plus_h) * 5 * np.cos(5 * psi))
    # w calculation
    w = (C0 * np.sqrt(g / k ** 3)) * k \
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
    dudt = (C0 * np.sqrt(g / k ** 3)) * (k * np.cos(theta)) \
        * (A11 * (epsilon) * np.cosh(k_z_plus_h) * omega * np.sin(psi)
           + A22 * (epsilon ** 2) * np.cosh(2 * k_z_plus_h) * 2 * omega * np.sin(2 * psi)
           + A31 * (epsilon ** 3) * np.cosh(k_z_plus_h) * omega * np.sin(psi)
           + A33 * (epsilon ** 3) * np.cosh(3 * k_z_plus_h) * 3 * omega * np.sin(3 * psi)
           + A42 * (epsilon ** 4) * np.cosh(2 * k_z_plus_h) * 2 * omega * np.sin(2 * psi)
           + A44 * (epsilon ** 4) * np.cosh(4 * k_z_plus_h) * 4 * omega * np.sin(4 * psi)
           + A51 * (epsilon ** 5) * np.cosh(k_z_plus_h) * omega * np.sin(psi)
           + A53 * (epsilon ** 5) * np.cosh(3 * k_z_plus_h) * 3 * omega * np.sin(3 * psi)
           + A55 * (epsilon ** 5) * np.cosh(5 * k_z_plus_h) * 5 * omega * np.sin(5 * psi))
    # dwdt vertical acceleration
    dwdt = (C0 * np.sqrt(g / k ** 3)) * k \
        * (A11 * epsilon * np.sinh(k_z_plus_h)*omega*-np.cos(psi)
           + A22 * (epsilon ** 2) * np.sinh(2 * k_z_plus_h) * 2 * omega * -np.cos(2 * psi)
           + A31 * (epsilon ** 3) * np.sinh(k_z_plus_h) * omega * -np.cos(psi)
           + A33 * (epsilon ** 3) * np.sinh(3 * k_z_plus_h) * 3 * omega * -np.cos(3 * psi)
           + A42 * (epsilon ** 4) * np.sinh(2 * k_z_plus_h) * 2 * omega * -np.cos(2 * psi)
           + A44 * (epsilon ** 4) * np.sinh(4 * k_z_plus_h) * 4 * omega * -np.cos(4 * psi)
           + A51 * (epsilon ** 5) * np.sinh(k_z_plus_h) * omega * - np.cos(psi)
           + A53 * (epsilon ** 5) * np.sinh(3 * k_z_plus_h) * 3 * omega * -np.cos(3 * psi)
           + A55 * (epsilon ** 5) * np.sinh(5 * k_z_plus_h) * 5 * omega * -np.cos(5 * psi))

    if z > eta:
        u = w = dudt = dwdt = 0

    return eta, u, w, dudt, dwdt

@dataclass
class Kinematics:
    """
    Returns:
        _type_: _description_
    """
    u
    du

@dataclass
class Linear(Kinematics):
    spectrum: Spectrum

    def fft_random_wave_sim(z_range: np.ndarray, d: np.ndarray, a: float, cond: bool):
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
        # np.random.seed(1234)

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

        eta = np.real(fftshift(fft(g1)))

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

            u_x[:, i_z] = np.real(fftshift(fft(g2))) * (z_init < eta)
            du_x[:, i_z] = np.real(fftshift(fft(g3))) * (z_init < eta)
            u_z[:, i_z] = np.real(fftshift(fft(g4))) * (z_init < eta)
            du_z[:, i_z] = np.real(fftshift(fft(g5))) * (z_init < eta)

        return eta, u_x, u_z, du_x, du_z




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

    # np.random.seed(1234)

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
    # np.random.seed(1234)

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

    eta = np.real(fftshift(fft(g1)))

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

        u_x[:, i_z] = np.real(fftshift(fft(g2))) * (z_init < eta)
        du_x[:, i_z] = np.real(fftshift(fft(g3))) * (z_init < eta)
        u_z[:, i_z] = np.real(fftshift(fft(g4))) * (z_init < eta)
        du_z[:, i_z] = np.real(fftshift(fft(g5))) * (z_init < eta)

    return eta, u_x, u_z, du_x, du_z


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

