import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

import airy as a1

def airy_wave_surface(k, A, xrange, yrange, omega, t, theta):

    xy = np.meshgrid(xrange, yrange)
    
    kx = k * np.cos(theta)
    ky = k * np.sin(theta)

    # kxy = [(kx * x, ky * y ) for (x,y) in xy]
    # kxx = np.array([x for (x,y) in kxy])
    # kyy = np.array([y for (x,y) in kxy])

    eta = A * np.sin(omega * t - xy * np.array([kx,ky]))
    eta.shape=(numx,numy)
    return eta

if __name__ == '__main__':


    h = 100
    T = 20
    A = 35/2
    theta = 0

    numx = 30
    numy = 30

    xrange = np.linspace(-5, 5, numx)
    yrange = np.linspace(-5, 5, numy)
    t=0

    k, omega = a1.airy_dispersion(h,T)

    eta = airy_wave_surface(k, A, xrange, yrange, omega, t, theta)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    surf = ax.plot_surface(xrange, yrange, eta)

    ax.set_zlim(-2,2)

    plt.show()