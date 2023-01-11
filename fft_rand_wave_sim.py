import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
import airy as arwave  # for morison eq function
import old_rand_wave_sim as rwave  # for JONSWAP
import rand_wave_spatial_sim as rws  # for dispersion relation


# below function isn't used - just to test fft version is working right
def ptws_random_wave_sim(t: float, z: float, d: float, om_range: np.ndarray, spctrl_dens: np.ndarray):
    """returns pointwave surface level eta and kinematics for x=0

    Args:
        t (float): time [s]
        z (float): height in water [m]
        d (float): water depth [m]
        om_range (np.ndarray): range of contributing angular frequencies [s^-1]
        spctrl_dens (np.ndarray): spectrum corresponding to om_range

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

    eta = np.sum(A * np.cos(om_range*t) + B * np.sin(om_range*t))

    z_init = z
    z = d * (d + z) / (d + eta) - d   # for Wheeler stretching

    k = np.empty(f_num)
    for i_om, om in enumerate(om_range):
        k[i_om] = rws.solve_dispersion(omega=om, h=d, upp=75)

    u_x = np.sum((A * np.cos(om_range*t) + B * np.sin(om_range*t)) * om_range * (np.cosh(k*(z+d))) / (np.sinh(k*d)))
    u_z = np.sum((-A * np.sin(om_range*t) + B * np.cos(om_range*t)) * om_range * (np.sinh(k*(z+d))) / (np.sinh(k*d)))

    du_x = np.sum((-A * np.sin(om_range*t) + B * np.cos(om_range*t)) * om_range**2 * (np.cosh(k*(z+d)))
                  / (np.sinh(k*d)))
    du_z = np.sum((-A * np.cos(om_range*t) - B * np.sin(om_range*t)) * om_range**2 * (np.sinh(k*(z+d)))
                  / (np.sinh(k*d)))

    if z_init > eta:
        u_x = u_z = du_x = du_z = 0

    return eta, u_x, u_z, du_x, du_z


def fft_random_wave_sim(z_range: np.ndarray, d: np.ndarray, om_range: np.ndarray, spctrl_dens: np.ndarray):
    """generates random wave surface and kinematics using FFT

    Args:
        z_range (np.ndarray): range of depths [m]
        d (float): water depth
        om_range (np.ndarray): range of angular velocities [s^-1]
        spctrl_dens (np.ndarray): spectrum corresponding to om_range

    Returns:
        eta (np.ndarray): wave surface height [m]
        u_x (np.ndarray): horizontal velociy at given z [ms^-1]
        u_v (np.ndarray): vertical velocity at given z [ms^-1]
        du_x (np.ndarray): horizontal acceleration at given z [ms^-2]
        du_v (np.ndarray): vertical acceleration at given z [ms^-2]
    """

    np.random.seed(1234)

    f_range = om_range / (2*np.pi)
    f_num = len(f_range)
    df = f_range[1] - f_range[0]

    A = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)
    B = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)

    i = complex(0, 1)
    g1 = A + B * i

    eta = np.real(fftshift(fft(g1)))

    k = np.empty(f_num)

    for i_f, f in enumerate(f_range):
        omega = 2 * np.pi * f
        k[i_f] = rws.solve_dispersion(omega, d, 75)

    u_x = np.empty((f_num, len(z_range)))
    du_x = np.empty((f_num, len(z_range)))
    u_z = np.empty((f_num, len(z_range)))
    du_z = np.empty((f_num, len(z_range)))

    z_num = len(z_range)
    z_ind = np.linspace(0, z_num-1, z_num)
    wheeler_z_range = np.linspace(-d, 0, z_num)

    for (i_z, z, w_z) in zip(z_ind, z_range, wheeler_z_range):

        i_z = int(i_z)

        g2 = (A+B*i) * 2*np.pi*f_range * (np.cosh(k*(w_z + d))) / (np.sinh(k*d))
        g3 = (B-A*i) * (2*np.pi*f_range)**2 * (np.cosh(k*(w_z+d))) / (np.sinh(k*d))
        g4 = (B-A*i) * (2*np.pi*f_range) * (np.sinh(k*(w_z+d))) / (np.sinh(k*d))
        g5 = (-A-B*i) * (2*np.pi*f_range)**2 * (np.sinh(k*(w_z+d))) / (np.sinh(k*d))

        u_x[:, i_z] = np.real(fftshift(fft(g2)))  # * (z < eta)
        du_x[:, i_z] = np.real(fftshift(fft(g3)))  # * (z < eta)
        u_z[:, i_z] = np.real(fftshift(fft(g4)))  # * (z < eta)
        du_z[:, i_z] = np.real(fftshift(fft(g5)))  # * (z < eta)

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


if __name__ == "__main__":

    # we will propagate a random wave and its kinematics at a fixed point x=0

    depth = 100.
    hs = 35.
    tp = 20.

    z_num = 150
    z_range = np.linspace(-depth, 50, z_num)
    dz = z_range[1] - z_range[0]

    # don't quite get this bit - for FFT to work
    freq = 1.00  # 3. / (2*np.pi)
    period = 100  # total time range
    nT = np.floor(period*freq)  # number of time points to evaluate
    t_num = int(nT)  # to work with rest of the code

    dt = 1/freq  # time step is determined by frequency
    t_range = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt  # centering time around 0

    f_range = np.linspace(1e-3, nT - 1, int(nT)) / (nT / freq)  # selecting frequency range from 0 to freq
    om_range = f_range * (2*np.pi)
    # all above taken from rand_wave_sim.py

    jnswp_dens = rwave.djonswap(f_range, hs, tp)

    eta_fft, u_x_fft, u_z_fft, du_x_fft, du_z_fft = fft_random_wave_sim(z_range, depth, om_range, jnswp_dens)

    F = np.empty((t_num, z_num))
    for i_t, t in enumerate(t_range):
        for i_z, z in enumerate(z_range):
            F[i_t, i_z] = arwave.morison_load(u_x_fft[i_t, i_z], du_x_fft[i_t, i_z])
    base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N

    z_grid, t_grid = np.meshgrid(z_range, t_range)

    # wheeler stretching (don't think this works)
    for i_t, t in enumerate(t_range):
        z_grid[i_t, :] = np.linspace(-depth, eta_fft[0][i_t], z_num)

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=u_x_fft.flatten())
    plt.plot(t_range, eta_fft[0], '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('u')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=u_z_fft.flatten())
    plt.plot(t_range, eta_fft[0], '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('v')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=du_x_fft.flatten())
    plt.plot(t_range, eta_fft[0], '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('du')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=du_z_fft.flatten())
    plt.plot(t_range, eta_fft[0], '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('dv')
    plt.colorbar()

    plt.figure()
    plt.plot(t_grid, base_shear)
    plt.ylabel('Force [MN]')
    plt.xlabel('Time')

    plt.show()

    # run below to show wheeler stretching for non-fft

    eta = np.empty(t_num)
    u_x = np.empty((t_num, z_num))
    u_z = np.empty((t_num, z_num))
    du_x = np.empty((t_num, z_num))
    du_z = np.empty((t_num, z_num))
    F = np.empty((t_num, z_num))

    for i_t, t in enumerate(t_range):
        for i_z, z in enumerate(z_range):
            eta[i_t], u_x[i_t, i_z], u_z[i_t, i_z], du_x[i_t, i_z], du_z[i_t, i_z] = ptws_random_wave_sim(t=t, z=z, d=depth, om_range=om_range, spctrl_dens=jnswp_dens)
            F[i_t, i_z] = arwave.morison_load(u_x[i_t, i_z], du_x[i_t, i_z])

    base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N

    z_grid, t_grid = np.meshgrid(z_range, t_range)

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=u_x.flatten())
    plt.ylim([-depth, 50])
    plt.plot(t_range, eta, '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('u')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=u_z.flatten())
    plt.plot(t_range, eta, '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('v')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=du_x.flatten())
    plt.plot(t_range, eta, '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('du')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=du_z.flatten())
    plt.plot(t_range, eta, '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('dv')
    plt.colorbar()

    plt.figure()
    plt.plot(t_grid, base_shear)
    plt.ylabel('Force [MN]')
    plt.xlabel('Time')

    plt.show()
