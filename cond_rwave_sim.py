import numpy as np
import matplotlib.pyplot as plt
import old_rand_wave_sim as rwave  # for JONSWAP
import fft_rand_wave_sim as nrwave  # for dispersion relation


def ptws_cond_rand_wave_sim(t: float, z:float, d:float, a: float, om_range: np.ndarray, spctrl_dens: np.ndarray):
    """returns a the sea surface level at time t and x=0 for a random wave sim conditioned on eta0=a

    Args:
        t (float): time [s]
        z (float): height in water [m]
        d (float): water depth [m]
        a (float): wave height at t=0 [m]
        om_range (np.ndarray): range of contributing angular frequencies [s^-1]
        spctrl_dens (np.ndarray): spectrum corresponding to om_range

    Returns:
        eta (float): sea level [m]
    """

    np.random.seed(1234)

    m = 0

    f_num = len(om_range)
    df = (om_range[1] - om_range[0]) / (2*np.pi)

    A = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df) * 0
    B = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df) * 0

    c = df * spctrl_dens
    d = df * spctrl_dens * om_range

    e = c * np.cos(om_range*t)
    f = d * np.sin(om_range*t)

    Q = (a - np.sum(A))/np.sum(c)
    R = (m - np.sum(om_range * A))/np.sum(d*om_range)

    A = A + Q * c
    B = B + R * d

    eta = np.sum(A * np.cos(om_range*t) + B * np.sin(om_range*t))

    z_init = z
    z = d * (d + z) / (d + eta) - d   # for Wheeler stretching

    k = np.empty(f_num)
    for i_om, om in enumerate(om_range):
        k[i_om] = nrwave.alt_solve_dispersion(omega=om, d=d)

    u_x = np.sum((A * np.cos(om_range*t) + B * np.sin(om_range*t)) * om_range * (np.cosh(k*(z+d))) / (np.sinh(k*d)))
    u_z = np.sum((-A * np.sin(om_range*t) + B * np.cos(om_range*t)) * om_range * (np.sinh(k*(z+d))) / (np.sinh(k*d)))

    du_x = np.sum((-A * np.sin(om_range*t) + B * np.cos(om_range*t)) * om_range**2 * (np.cosh(k*(z+d)))
                  / (np.sinh(k*d)))
    du_z = np.sum((-A * np.cos(om_range*t) - B * np.sin(om_range*t)) * om_range**2 * (np.sinh(k*(z+d)))
                  / (np.sinh(k*d)))

    if z_init > eta:
        u_x = u_z = du_x = du_z = 0

    return eta, u_x, u_z, du_x, du_z


if __name__ == "__main__":

    hs = 10.
    tp = 12.
    a = 20.

    t_num = 200
    t_range = np.linspace(-50, 50, t_num)

    om_num = 50
    om_range = np.linspace(start=1e-1, stop=3, num=om_num)

    f_range = om_range / (np.pi * 2)
    jnswp_dens = rwave.djonswap(f_range, hs, tp)

    eta = np.empty(t_num)
    eta_cond = np.empty(t_num)

    for i_t, t in enumerate(t_range):
        eta[i_t], eta_cond[i_t] = ptws_cond_rand_wave_sim(t=t, a=a, om_range=om_range, spctrl_dens=jnswp_dens)

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