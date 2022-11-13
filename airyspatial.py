import matplotlib.pyplot as plt
import numpy as np


import airy as a1

def airy_wave_surface(k, A, xrange, yrange, omega, t, theta):
    
    kx = k * np.cos(theta)
    ky = k * np.sin(theta)

    xpoints, ypoints = np.meshgrid(xrange, yrange)

    omega = np.tile(omega, numx * numy).reshape(numx, numy)
    eta = A * np.sin(omega * t - kx * xpoints - ky * ypoints)

    eta.shape = (numx, numy)
    
    return eta

if __name__ == '__main__':


    h = 100
    T = 20
    A = 35/2
    theta = np.pi/4

    numx = 30
    numy = 30

    xrange = np.linspace(-500, 500, numx)
    yrange = np.linspace(-500, 500, numy)
    t = 10

    k, omega = a1.airy_dispersion(h, T)

    eta = airy_wave_surface(k, A, xrange, yrange, omega, t, theta)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    X, Y = np.meshgrid(xrange, yrange)
    surf = ax.plot_surface(X, Y, eta)

    

   # plt.contourf(xrange, yrange, eta)


    plt.show()
