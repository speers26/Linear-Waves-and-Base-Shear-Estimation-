import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def djonswap(f, hs, tp):
    """
    returns JONSWAP density for given frequency 

    Args:
        f (_type_): frequency [s^-1]
        hs (_type_): significant wave height [m]
        tp (_type_): significant wave period [s]
    """
    g = 9.81
    fp = 1. / tp

    ## from https://wikiwaves.org/Ocean-Wave_Spectra
    gamma = 3.3
    sigma_a = 0.07 
    sigma_b = 0.09

    sigma = (f < fp) *  sigma_a + (f >= fp) * sigma_b

    gamma_coeff = gamma ** np.exp(-0.5 * (((f / fp - 1)/sigma) ** 2)) 
    dens = g ** 2 * (2 * np.pi) ** -4 * f ** -5 * np.exp(-1.25 * (tp*f) ** -4) * gamma_coeff
    
    area = sum(dens*df)

    dens *= hs ** 2 / (16 * area)

    return dens

def random_waves_surface(f,t):

    A = np.random.normal(0, 1, size=(1,n_freq)) *  np.sqrt(dens*df) 
    B = np.random.normal(0, 1, size=(1,n_freq)) *  np.sqrt(dens*df) 
    outer_tf = np.outer(t,f) 
    eta = np.sum(A * np.cos(2*np.pi*outer_tf) + B * np.sin(2*np.pi*outer_tf), axis=1)

    return eta

if __name__ == "__main__":

    hs = 35
    tp = 10
    f_p = 1/tp

    n_freq = 500
    f_seq = np.linspace(1e-3,f_p*5,n_freq)
    df = f_seq[1] - f_seq[0]

    dens = djonswap(f_seq, hs, tp)
    area = sum(dens * df)

    spectral_estHs = 4 * np.sqrt(area)

    n_seconds = 100
    freq = 4
    n_time = freq*n_seconds
    time = np.linspace(0,n_seconds,n_time)
        
    eta = random_waves_surface(f_seq,time)

    surface_estHs = 4 * np.std(eta)

    print(spectral_estHs,surface_estHs)
    

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(f_seq,dens)
    
    plt.subplot(2,1,2)
    plt.plot(time,eta)
    plt.show()