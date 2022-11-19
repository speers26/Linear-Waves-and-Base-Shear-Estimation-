import numpy as np
import matplotlib.pyplot as plt
import rand_wave_sim as r1

def bm_wrpd_Gaus(omega:np.ndarray, phi:np.ndarray, alpha:np.ndarray, om_p:np.ndarray, gamma:np.ndarray,
                    r:np.ndarray, phi_m:np.ndarray, beta:np.ndarray, nu:np.ndarray, sig_l:np.ndarray, sig_r:np.ndarray):
    """returns spreading function D(omega, phi)

    Args:
        omega (np.ndarray): angular freq
        phi (np.ndarray): direction (from)
        alpha (np.ndarray): scaling parameter
        om_p (np.ndarray): peak freq
        gamma (np.ndarray): peak enhancement factor
        r (np.ndarray): spectral tail decay index
        phi_m (np.ndarray): mean direction
        beta (np.ndarray): limiting peak separation
        nu (np.ndarray): peak separation shape
        sig_l (np.ndarray): limiting angular width
        sig_r (np.ndarray): angular width shape
    """

    