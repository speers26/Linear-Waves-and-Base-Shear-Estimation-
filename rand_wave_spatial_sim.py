import numpy as np
import matplotlib.pyplot as plt

def frq_dr_spctrm(omega:np.ndarray, phi:np.ndarray, alpha:np.ndarray, om_p:np.ndarray, gamma:np.ndarray,
                    r:np.ndarray, phi_m:np.ndarray, beta:np.ndarray, nu:np.ndarray, sig_l:np.ndarray, sig_r:np.ndarray):
    """returns frequency direction spectrum

    Args:
        omega (np.ndarray): angular frequency
        phi (np.ndarray): _description_
        alpha (np.ndarray): _description_
        om_p (np.ndarray): _description_
        gamma (np.ndarray): _description_
        r (np.ndarray): _description_
        phi_m (np.ndarray): _description_
        beta (np.ndarray): _description_
        nu (np.ndarray): _description_
        sig_l (np.ndarray): _description_
        sig_r (np.ndarray): _description_
    """
    dens = sprd_fnc(omega, phi, om_p, phi_m, beta, nu, sig_l, sig_r) * d_jonswap(omega, alpha, om_p, gamma, r)

    return dens

def sprd_fnc(omega:np.ndarray, phi:np.ndarray, om_p:np.ndarray, phi_m:np.ndarray, beta:np.ndarray, nu:np.ndarray, sig_l:np.ndarray, sig_r:np.ndarray):
    """returns bimodal wrapped Gaussian spreading function D(omega, phi)

    Args:
        omega (np.ndarray): angular frequency
        phi (np.ndarray): direction (from)
        om_p (np.ndarray): peak ang freq
        phi_m (np.ndarray): mean direction
        beta (np.ndarray): limiting peak separation
        nu (np.ndarray): peak separation shape
        sig_l (np.ndarray): limiting angular width
        sig_r (np.ndarray): angular width shape
    """
    k_num = 200
    k_range = np.linspace(start = -k_num/2, stop = k_num/2, num = k_num + 1)

    phi_m1 = phi_m + beta * np.exp(-nu * min(om_p / np.abs(omega), 1)) / 2
    phi_m2 = phi_m - beta * np.exp(-nu * min(om_p / np.abs(omega), 1)) / 2
    phi_arr = np.array([phi_m1, phi_m2])

    sigma = sig_l - sig_r / 3 * (4 * (om_p / np.abs(omega)) ** 2 - (om_p / np.abs(omega)) ** 8)

    nrm_cnst = (2 * sigma * np.sqrt(2 * np.pi)) ** -1
    dens_k = np.empty(k_num + 1)

    for i_k, k in enumerate(k_range):
        exp_term = np.exp( -0.5 * ((phi - phi_arr - 2 * np.pi * k) / sigma) ** 2)
        dens_k[i_k] = np.sum(exp_term)

    dens = nrm_cnst * np.sum(dens_k)

    return dens

def d_jonswap(omega:np.ndarray, alpha:np.ndarray, om_p:np.ndarray, gamma:np.ndarray, r:np.ndarray):
    """jonswap density using formulation used in Jake's paper

    Args:
        omega (np.ndarray): angular frequency
        alpha (np.ndarray): scaling parameter
        om_p (np.ndarray): peak ang freq
        gamma (np.ndarray): peak enhancement factor
        r (np.ndarray): spectral tail decay index
    """

    delta = np.exp( -(2 * (0.07 + 0.02 * (om_p > np.abs(omega)) )) ** -2 * (np.abs(omega) / om_p - 1) ** 2)

    dens = alpha * omega ** -r * np.exp( -r / 4 * (np.abs(omega) / om_p ) ** -4) * gamma ** delta

    return dens


if __name__ == '__main__':

    ### pars set accoring to 'classic example' given in 
    # https://www.mendeley.com/reference-manager/reader/6c295827-d975-39e4-ad43-c73f0f51b060/21c9456c-b9ef-e1bb-1d36-7c1780658222
    alpha = 0.7
    om_p = 0.8
    gamma = 3.3
    r = 5
    phi_m = np.pi / 2
    beta = 4
    nu = 2.7
    sig_l = 0.55
    sig_r = 0.26

    om_num = 100
    om_range = np.linspace(start = 1e-3, stop = 2 * np.pi, num = om_num)

    phi_num = 100
    phi_range = np.linspace(start = 0, stop = 3, num = phi_num)

    # ### plotting contours

    D_sprd = np.empty((om_num, phi_num))
    for i_m, om in enumerate(om_range):
        for i_p, phi in enumerate(phi_range):
            D_sprd[i_m, i_p] = sprd_fnc(om, phi, om_p, phi_m, beta, nu, sig_l, sig_r)

    d_om = om_range[1] - om_range[0]
    d_phi = phi_range[1] - phi_range[0]

    sprd_vol = sum(d_om * d_phi * D_sprd)
    print(sprd_vol)

    jnswp_dns = np.empty(om_num)
    for i_o, om in enumerate(om_range):
        jnswp_dns[i_o] = d_jonswap(om, alpha, om_p, gamma, r)

    jnswp_area = sum(d_om * jnswp_dns)
    print(jnswp_area)

    Dr_spctrm = np.empty((om_num, phi_num))
    for i_m, om in enumerate(om_range):
        for i_p, phi in enumerate(phi_range):
            Dr_spctrm[i_m, i_p] = frq_dr_spctrm(om, phi, alpha, om_p, gamma, r, phi_m, beta, nu, sig_l, sig_r)

    spctrm_vol = sum(sum(d_om * d_phi * Dr_spctrm))
    print(spctrm_vol)

    OM, PHI = np.meshgrid(om_range, phi_range)

    plt.figure()

    plt.subplot(1,3,1)
    plt.contour(PHI, OM, Dr_spctrm, levels = 20)

    plt.subplot(1,3,2)
    plt.plot(om_range, jnswp_dns)

    plt.subplot(1,3,3)
    plt.contour(PHI, OM, D_sprd, levels = 20)

    plt.show()
