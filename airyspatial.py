import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import airy as a1

def airy_wave_surface(k, A, xrange, yrange, omega, t, theta):
    
    kx = k * np.cos(theta)
    ky = k * np.sin(theta)

    kxky = [(kx * x, ky * y) for x in xrange for y in yrange]
    kxx = np.array([x for (x,y) in kxky])
    kyy = np.array([y for (x,y) in kxky])

    omega = np.tile(omega, numx * numy)
    eta = A * np.sin(omega * t - kxx - kyy)

    eta.shape = (numx, numy)
    
    return eta

if __name__ == '__main__':


    h = 100
    T = 20
    A = 35/2
    theta = np.pi /4

    numx = 30
    numy = 30

    xrange = np.linspace(-5, 5, numx)
    yrange = np.linspace(-5, 5, numy)
    t = 0

    k, omega = a1.airy_dispersion(h, T)

    eta = airy_wave_surface(k, A, xrange, yrange, omega, t, theta)
    print(eta)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y = np.meshgrid(xrange, yrange)
    surf = ax.plot_surface(X, Y, eta)

    ax.set_zlim(-1.5,1.5)

    plt.show()
