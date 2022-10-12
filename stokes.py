import numpy as np


def stokes_kinematics(k: np.ndarray, h: np.ndarray, A: np.ndarray, x: np.ndarray,
                      omega: np.ndarray, t: np.ndarray, theta: np.array, z: np.ndarray):
    """ Generate Stokes kinematic profiles

    Args:
        k (np.ndarray):  wave number (coming from Stokes disp.)
        h (np.ndarray): wave depth TODO: rename to d
        A (np.ndarray):  wave amplitude: A = H/2 -> you will get an asymmetric profile
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
    Aco = np.empty(8)
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
    Bco[0] = (1 / np.tanh(kd)) * (1 + 2 * S) / (2 * (1-S))
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
           + A51 * (epsilon ** 5) * np.cosh(k_z_plus_h)) * np.cos(psi) \
        + A53 * (epsilon ** 5) * np.cosh(3 * k_z_plus_h) * 3 * np.cos(3 * psi) \
        + A55 * (epsilon ** 5) * np.cosh(5 * k_z_plus_h) * 5 * np.cos(5 * psi)
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

    return eta, u, w, dudt, dwdt


def fDispersionSTOKES5(h, H1, T):
    """
    Solves the progressive wave dispersion equation

    Args:
        h (np.ndarray): depth [m]
        H1 (np.ndarray):  wave height [m]
        T (np.ndarray): wave period

    Returns:
        np.array: wave number k [1/m]
    """

    g = 9.81
    omega = 2 * np.pi / T

    p = omega ** 2 * h / g
    q = (np.tanh(p ** 0.75)) ** (-2 / 3)
    k0 = (4 * np.pi ** 2) / (g * T ** 2)

    #TODO: fix fzero
    [k , F] = fzero(@(k) 1+(H1**2*k**2)/8+(H1**4*k**4)/128-omega/((9.81*k)**0.5),k0)

    return k