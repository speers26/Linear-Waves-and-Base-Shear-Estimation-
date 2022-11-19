import matplotlib.pyplot as plt
import numpy as np
import os
import imageio

import airy as a1

def airy_wave_surface(k, A, xrange, yrange, omega, t, theta):
    
    kx = k * np.cos(theta)
    ky = k * np.sin(theta)

    xpoints, ypoints = np.meshgrid(xrange, yrange)

    omega = np.tile(omega, numx * numy).reshape(numy, numx)
    psi = omega * t - kx * xpoints - ky * ypoints
    eta = A * np.sin(omega * t - kx * xpoints - ky * ypoints)

    eta.shape = (numy, numx)
    
    return eta

if __name__ == '__main__':


    h = 100
    T = 20
    A = 35/2
    theta = 0
    k, omega = a1.airy_dispersion(h, T)

    numx = 30
    numy = 30

    xrange = np.linspace(-100, 100, numx)
    yrange = np.linspace(-500, 500, numy)
    X, Y = np.meshgrid(xrange, yrange)

    nt = 100
    trange = np.linspace(0,100,nt)
    names = []

    for t in trange:
        eta = airy_wave_surface(k, A, xrange, yrange, omega, t, theta)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        surf = ax.plot_surface(X, Y, eta)

        name = f'time_{t}.png'
        names.append(name)

        plt.savefig(name)
        plt.close()

    with imageio.get_writer('airymoving.gif', mode='I') as writer:
        for filename in names:
            image = imageio.imread(filename)
            writer.append_data(image)

    for name in set(names):
        os.remove(name)





