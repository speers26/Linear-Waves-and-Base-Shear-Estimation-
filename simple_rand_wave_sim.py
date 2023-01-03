import numpy as np
import matplotlib.pyplot as plt
import rand_wave_sim as rwave  # for JONSWAP
import rand_wave_spatial_sim as rws  # for dispersion relation


def ptws_random_wave_sim(t: float, z: float, d: float, om_range: np.ndarray, spctrl_dens: np.ndarray):
    """returns pointwave surface level eta and kinematics for x=0

    Args:
        t (float): time [s]
        z (float): height in water [m]
        d (float): water depth [m]
        om_range (np.ndarray): range of contributing angular frequencies [s^-1]
        spctrl_dens (np.ndarray): spectrum corresponding to om_range

    Returns:
        eta (float):
        u_x (float):
    """

    z_init = z

    if z > 0:  # kinematic stretching
        z = 0

    np.random.seed(1)

    f_num = len(om_range)
    df = (om_range[1] - om_range[0]) / (2*np.pi)

    A = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)
    B = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)

    eta = np.sum(A * np.cos(om_range*t) + B * np.sin(om_range*t))

    k = np.empty(f_num)
    for i_om, om in enumerate(om_range):
        k[i_om] = rws.solve_dispersion(omega=om, h=d, upp=1)

    u_x = np.sum((A * np.cos(om_range*t) + B * np.sin(om_range*t)) * om_range * (np.cosh(k*(z+d))) / (np.sinh(k*d)))

    if z_init > eta:
        u_x = 0

    return eta, u_x


if __name__ == "__main__":

    # we will propagate a random wave and its kinematics at a fixed point x=0

    depth = 100.
    hs = 30.
    tp = 10.

    t_num = 200
    t_range = np.linspace(-50, 50, t_num)

    z_num = 150
    z_range = np.linspace(-depth, 50, z_num)

    om_num = 50
    om_range = np.linspace(start=1e-3, stop=3, num=om_num)

    f_range = om_range / (np.pi * 2)
    jnswp_dens = rwave.djonswap(f_range, hs, tp)

    eta = np.empty(t_num)
    u_x = np.empty((t_num, z_num))

    for i_t, t in enumerate(t_range):
        for i_z, z in enumerate(z_range):
            eta[i_t], u_x[i_t, i_z] = ptws_random_wave_sim(t=t, z=z, d=depth, om_range=om_range, spctrl_dens=jnswp_dens)

    print(np.std(eta)*4)

    z_grid, t_grid = np.meshgrid(z_range, t_range)

    plt.figure()
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=u_x.flatten())
    plt.plot(t_range, eta, '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.colorbar()
    plt.show()
