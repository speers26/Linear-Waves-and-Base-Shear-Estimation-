import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt


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


def airy_dispersion(h: np.ndarray, T: np.ndarray):
    """solves dispersion relation for wave number

    Args:
        h (np.ndarray): water depth
        T (np.ndarray): period [s]
    """

    omega = 2 * np.pi / T

    f = lambda k: dispersion_diff(k, h, omega)

    k = optimize.bisect(f, 1e-7, 1)

    return k, omega


def dispersion_diff(k: np.ndarray, h: np.ndarray, omega: np.ndarray):
    """function to optimise in airy_dispersion

    Args:
        k (np.ndarray): wave number
        h (np.ndarray): water depth
        omega (np.ndarray): angular frequency
    """
    g = 9.81
    return omega ** 2 - g * k * np.tanh(k * h)


def morison_load(u, du, diameter=1.0, rho=1024.0, c_m=1.0, c_d=1.0):
    """compute unit Morison load for a vertical cylinder

    Args:
        u (np.ndarray): horizontal velocity [m/s]
        du (np.ndarray): horizontal acceleration [m/s^2]
        diameter (float, optional): _description_. Defaults to 1.0. [m]
        rho (float, optional): _description_. Defaults to 1024.0. [kg/m^3]
        c_m (float, optional): _description_. Defaults to 1.0. [unitless]
        c_d (float, optional): _description_. Defaults to 1.0. [unitless]

    Returns:
        np.ndarray: horizontal unit morrison load [N/m]
    """

    return rho * c_m * (np.pi / 4) * (diameter ** 2) * du + 0.5 * rho * c_d * diameter * u * np.abs(u)


if __name__ == '__main__':

    h = 100
    T = 20
    A = 35/2

    x = 0
    n_depth = 151
    z_range = np.linspace(-h, 50, n_depth)
    dz = z_range[1] - z_range[0]

    n_time = 200
    time = np.linspace(-20, 20, 200)

    eta = np.empty(n_time)
    u = np.empty((n_time, n_depth))
    w = np.empty((n_time, n_depth))
    du = np.empty((n_time, n_depth))
    dw = np.empty((n_time, n_depth))
    F = np.empty((n_time, n_depth))

    k, omega = airy_dispersion(h, T)

    for i_t, t in enumerate(time):
        for i_z, z in enumerate(z_range):
            eta[i_t], u[i_t, i_z], w[i_t, i_z], du[i_t, i_z], dw[i_t, i_z] = airy_kinematics(k, h, A, x, omega, t, z)

            F[i_t, i_z] = morison_load(u[i_t, i_z], du[i_t, i_z])

    base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N

    plt.figure()
    plt.subplot(2, 2, 1)
    z_range_grid, time_grid = np.meshgrid(z_range, time)

    plt.scatter(time_grid.flatten(), z_range_grid.flatten(), s=1, c=u.flatten())
    plt.plot(time, eta, '-k')
    plt.title('u')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.scatter(time_grid.flatten(), z_range_grid.flatten(), s=1, c=w.flatten())
    plt.plot(time, eta, '-k')
    plt.title('w')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.scatter(time_grid.flatten(), z_range_grid.flatten(), s=1, c=du.flatten())
    plt.plot(time, eta, '-k')
    plt.title('du')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.scatter(time_grid.flatten(), z_range_grid.flatten(), s=1, c=dw.flatten())
    plt.plot(time, eta, '-k')
    plt.title('dw')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.colorbar()

    plt.figure()
    plt.plot(time_grid, base_shear)
    plt.ylabel('Force [MN]')
    plt.xlabel('Time')

    plt.show()
