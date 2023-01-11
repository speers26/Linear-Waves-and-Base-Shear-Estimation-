import numpy as np
import matplotlib.pyplot as plt
import old_rand_wave_sim as rwave  # for JONSWAP


def ptws_cond_rand_wave_sim(t: float, a: float, om_range: np.ndarray, spctrl_dens: np.ndarray):
    """returns a the sea surface level at time t and x=0 for a random wave sim conditioned on eta0=a

    Args:
        t (float): time [s]
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

    A = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)
    B = np.random.normal(0, 1, size=(1, f_num)) * np.sqrt(spctrl_dens*df)

    c = df * spctrl_dens
    d = df * spctrl_dens * om_range

    e = c * np.cos(om_range*t)
    f = d * np.sin(om_range*t)

    Q = (a - np.sum(A))/np.sum(c)
    R = (m - np.sum(om_range * A))/np.sum(d*om_range)

    eta = np.sum(A * np.cos(om_range*t) + B * np.sin(om_range*t))
    eta_cond = np.sum(A * np.cos(om_range*t) + B * np.sin(om_range*t) + Q * e + R * f)

    return eta, eta_cond


if __name__ == "__main__":

    hs = 5.
    tp = 5.
    a = 5

    t_num = 200
    t_range = np.linspace(-25, 25, t_num)

    om_num = 50
    om_range = np.linspace(start=1e-1, stop=3, num=om_num)

    f_range = om_range / (np.pi * 2)
    jnswp_dens = rwave.djonswap(f_range, hs, tp)

    eta = np.empty(t_num)
    eta_cond = np.empty(t_num)

    for i_t, t in enumerate(t_range):
        eta[i_t], eta_cond[i_t] = ptws_cond_rand_wave_sim(t=t, a=a, om_range=om_range, spctrl_dens=jnswp_dens)

    plt.figure()
    plt.plot(t_range, eta, '-k')
    plt.plot(t_range, eta_cond, '--r')
    plt.show()
