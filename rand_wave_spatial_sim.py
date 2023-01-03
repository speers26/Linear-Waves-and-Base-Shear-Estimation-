import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import imageio


def random_wave_surface_and_kinematics(om_range: np.ndarray, phi_range: np.ndarray, t: np.ndarray, x_range: np.ndarray,
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
    dens = sprd_fnc(omega, phi, om_p, phi_m, beta, nu, sig_l, sig_r) * d_jonswap(omega, alpha, om_p, gamma, r)

    return dens


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


def d_jonswap(omega: float, alpha: float, om_p: float, gamma: float, r: float):
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


if __name__ == '__main__':

    depth = 100.
    hs = 30.

    # pars set accoring to 'classic example' given in
    # https://www.mendeley.com/reference-manager/reader/6c295827-d975-39e4-ad43-c73f0f51b060/21c9456c-b9ef-e1bb-1d36-7c1780658222
    # play with these to change form of wave surface
    alpha = 0.7
    om_p = 0.8
    gamma = 3.3  # make larger to decrease width of Jonswap
    r = 5.
    phi_m = np.pi
    beta = 4.
    nu = 2.7
    sig_l = 0.55  # make smaller to decrease directional spreading
    sig_r = 0.26  # make zero to decrease directional spreading

    om_num = 50
    om_range = np.linspace(start=1e-3, stop=3, num=om_num)

    phi_num = 100
    phi_range = np.linspace(start=0, stop=2 * np.pi, num=phi_num)

    x_num = 100
    x_range = np.linspace(start=-50, stop=50, num=x_num)

    y_num = 100
    y_range = np.linspace(start=-50, stop=50, num=y_num)

    d_om = om_range[1] - om_range[0]
    d_phi = phi_range[1] - phi_range[0]

    # plotting contours

    D_sprd = np.empty((phi_num, om_num))
    for i_o, om in enumerate(om_range):
        for i_p, phi in enumerate(phi_range):
            D_sprd[i_p, i_o] = sprd_fnc(om, phi, om_p, phi_m, beta, nu, sig_l, sig_r)

    jnswp_dns = np.empty(om_num)
    for i_o, om in enumerate(om_range):
        jnswp_dns[i_o] = d_jonswap(om, alpha, om_p, gamma, r)

    jnswp_area = sum(d_om * jnswp_dns)
    jnswp_dns *= hs ** 2 / (16 * jnswp_area)  # rescale to provide given hs
    jnswp_area = sum(d_om * jnswp_dns)

    print(jnswp_area)

    Dr_spctrm = np.empty((phi_num, om_num))
    for i_o, om in enumerate(om_range):
        for i_p, phi in enumerate(phi_range):
            Dr_spctrm[i_p, i_o] = frq_dr_spctrm(om, phi, alpha, om_p, gamma, r, phi_m, beta, nu, sig_l, sig_r)

    spctrm_vol = sum(sum(d_om * d_phi * Dr_spctrm))

    Dr_spctrm *= hs ** 2 / (16 * spctrm_vol)  # rescale to provide given hs
    spctrm_vol = sum(sum(d_om * d_phi * Dr_spctrm))

    print(spctrm_vol)

    omega_grid, phi_grid = np.meshgrid(om_range, phi_range)  # expand omega and phi axis onto a grid to plot contours

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.contour(omega_grid, phi_grid, Dr_spctrm, levels=[15, 30, 60, 90, 120, 150, 180, 210, 240])
    plt.xlabel("angular freq")
    plt.ylabel("direction")

    plt.subplot(1, 3, 2)
    plt.plot(om_range, jnswp_dns)
    plt.xlabel("angular freq")
    plt.ylabel("density")

    plt.subplot(1, 3, 3)
    plt.contour(omega_grid, phi_grid, D_sprd, levels=20)
    plt.xlabel("angular freq")
    plt.ylabel("direction")

    plt.show()

    nt = 100
    trange = np.linspace(0, 15, nt)
    names = []
    x_grid, y_grid = np.meshgrid(x_range, y_range)  # expand x and y axis onto a grid to plot eta over

    for it, t in enumerate(trange):

        eta = random_wave_surface_and_kinematics(om_range, phi_range, t, x_range, y_range, depth)

        print(np.var(eta))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        surf = ax.plot_surface(x_grid, y_grid, eta)

        ax.set_zlim(-40, 40)

        name = f'time_{it}.png'
        names.append(name)

        plt.savefig(name)
        plt.close()

    with imageio.get_writer('random-waves-moving.gif', mode='I') as writer:
        for filename in names:
            image = imageio.imread(filename)
            writer.append_data(image)

    for name in set(names):
        os.remove(name)
