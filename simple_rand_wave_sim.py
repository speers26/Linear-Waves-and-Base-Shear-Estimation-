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

    np.random.seed(1)

    f_num = len(om_range)
    df = (om_range[1] - om_range[0]) / (2*np.pi)

    A = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)
    B = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)

    eta = np.sum(A * np.cos(om_range*t) + B * np.sin(om_range*t))

    return eta


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

    for i_t, t in enumerate(t_range):
        z = 0
        eta[i_t] = ptws_random_wave_sim(t=t, z=0, d=depth, om_range=om_range, spctrl_dens=jnswp_dens)

    print(np.std(eta)*4)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(om_range, jnswp_dens / 2*np.pi)

    plt.subplot(2, 1, 2)
    plt.plot(t_range, eta)
    plt.show()
