import numpy as np
import matplotlib.pyplot as plt
import old_rand_wave_sim as rwave  # for JONSWAP
import rand_wave_spatial_sim as rws  # for dispersion relation
import airy as arwave  # for morison eq function


def ptws_cond_rand_wave_sim(t: float, z: float, depth: float, a: float, om_range: np.ndarray, spctrl_dens: np.ndarray):
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
    #  this is essentially the same as the pointwise non fft code in fft_rand_wave_sim.py - should really merge these
    np.random.seed(1234)

    m = 0

    f_num = len(om_range)
    df = (om_range[1] - om_range[0]) / (2*np.pi)

    A = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)
    B = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)

    c = df * spctrl_dens
    d = df * spctrl_dens * om_range

    #  e = c * np.cos(om_range*t)  # not needed if you adjust A
    #  f = d * np.sin(om_range*t)  # not needed if you adjust B

    Q = (a - np.sum(A))/np.sum(c)
    R = (m - np.sum(om_range * A))/np.sum(d*om_range)

    A = A + Q * c
    B = B + R * d

    eta = np.sum(A * np.cos(om_range*t) + B * np.sin(om_range*t))

    z_init = z
    z = depth * (depth + z) / (depth + eta) - depth   # for Wheeler stretching

    k = np.empty(f_num)
    for i_om, om in enumerate(om_range):
        k[i_om] = rws.solve_dispersion(omega=om, h=depth, upp=75)

    u_x = np.sum((A * np.cos(om_range*t) + B * np.sin(om_range*t)) * om_range * (np.cosh(k*(z+depth))) / (np.sinh(k*depth)))
    u_z = np.sum((-A * np.sin(om_range*t) + B * np.cos(om_range*t)) * om_range * (np.sinh(k*(z+depth))) / (np.sinh(k*depth)))

    du_x = np.sum((-A * np.sin(om_range*t) + B * np.cos(om_range*t)) * om_range**2 * (np.cosh(k*(z+depth)))
                  / (np.sinh(k*depth)))
    du_z = np.sum((-A * np.cos(om_range*t) - B * np.sin(om_range*t)) * om_range**2 * (np.sinh(k*(z+depth)))
                  / (np.sinh(k*depth)))

    if z_init > eta:
        u_x = u_z = du_x = du_z = 0

    return eta, u_x, u_z, du_x, du_z


if __name__ == "__main__":

    hs = 10.
    tp = 12.
    a = 20.
    depth = 100

    z_num = 150
    z_range = np.linspace(-depth, 50, z_num)
    dz = z_range[1] - z_range[0]

    t_num = 200
    t_range = np.linspace(-50, 50, t_num)

    om_num = 50
    om_range = np.linspace(start=1e-1, stop=3, num=om_num)

    f_range = om_range / (np.pi * 2)
    jnswp_dens = rwave.djonswap(f_range, hs, tp)

    eta = np.empty(t_num)
    u_x = np.empty((t_num, z_num))
    u_z = np.empty((t_num, z_num))
    du_x = np.empty((t_num, z_num))
    du_z = np.empty((t_num, z_num))
    F = np.empty((t_num, z_num))

    for i_t, t in enumerate(t_range):
        for i_z, z in enumerate(z_range):
            eta[i_t], u_x[i_t, i_z], u_z[i_t, i_z], du_x[i_t, i_z], du_z[i_t, i_z] = ptws_cond_rand_wave_sim(t=t, z=z, depth=depth, a=a, om_range=om_range, spctrl_dens=jnswp_dens)
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
