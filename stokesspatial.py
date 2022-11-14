import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
import stokes as sk1
import os
import imageio

def stokes_wave_surface(k, h, A, xrange, yrange, omega, t, theta):

  
    kx = k * np.cos(theta)
    ky = k * np.sin(theta)

    xpoints, ypoints = np.meshgrid(xrange, yrange)

    omega = np.tile(omega, numx * numy).reshape(numx, numy)

    # kd = kx * h + ky * h
    kd = k * h #TODO

    S = 1 / np.cosh(2 * kd)
    # Calculation of the A coefficients
    Aco = np.empty(9)
    Aco[0] = 1 / np.sinh(kd)
    Aco[1] = 3 * (S ** 2) / (2 * ((1. - S) ** 2))
    Aco[2] = (-4 - 20 * S + 10 * (S ** 2) - 13 * (S ** 3)) / (8 * np.sinh(kd) * ((1 - S) ** 3))
    Aco[3] = (-2 * (S ** 2) + 11 * (S ** 3)) / (8 * np.sinh(kd) * ((1.-S) ** 3))
    Aco[4] = (12 * S - 14 * (S ** 2) - 264 * (S ** 3) - 45 * (S ** 4) - 13 * (S ** 5)) / (24*((1.-S)**5))
    Aco[5] = (10 * (S ** 3) - 174 * (S ** 4) + 291 * (S ** 5) + 278 * (S ** 6)) / (48 * (3 + 2 * S) * ((1 - S) ** 5))
    Aco[6] = (-1184 + 32 * S + 13232 * (S ** 2) + 21712 * (S ** 3) + 20940 * (S ** 4) + 12554 * (S ** 5) - 500 *
              (S ** 6) - 3341 * (S ** 7) - 670 * (S ** 8)) / (64 * np.sinh(kd) * (3 + 2 * S) * (4 + S) * ((1 - S) ** 6))
    Aco[7] = (4 * S + 105 * (S ** 2) + 198 * (S ** 3) - 1376 * (S ** 4) - 1302 * (S ** 5) - 117 * (S ** 6) +
              58 * (S ** 7))/(32 * np.sinh(kd) * (3 + 2 * S) * ((1 - S) ** 6))
    Aco[8] = (-6 * (S ** 3) + 272 * (S ** 4) - 1552 * (S ** 5) + 852 * (S ** 6) + 2029 * (S ** 7) + 430 * (S ** 8)) \
        / (64 * np.sinh(kd) * (3 + 2 * S) * (4 + S) * ((1 - S) ** 6))
    # Calculation of the B coefficients
    Bco = np.empty(6)
    Bco[0] = (1 / np.tanh(kd)) * (1 + 2 * S) / (2 * (1 - S))
    Bco[1] = -3 * (1 + 3 * S + 3 * (S ** 2) + 2 * (S ** 3)) / (8 * ((1 - S) ** 3))
    Bco[2] = (1 / np.tanh(kd)) * (6 - 26 * S - 182 * (S ** 2) - 204 * (S ** 3) - 25 * (S ** 4) + 26 * (S ** 5)) \
        / (6 * (3 + 2 * S) * ((1 - S) ** 4))
    Bco[3] = (1./np.tanh(kd)) * (24 + 92 * S + 122 * (S ** 2) + 66 * (S ** 3) + 67 * (S ** 4) + 34 * (S ** 5)) \
        / (24 * (3 + 2 * S) * ((1 - S) ** 4))
    Bco[4] = 9 * (132 + 17 * S - 2216 * (S ** 2) - 5897 * (S ** 3) - 6292 * (S ** 4) - 2687 * (S ** 5)
                  + 194 * (S ** 6) + 467 * (S ** 7) + 82 * (S ** 8)) / (128 * (3 + 2 * S) * (4 + S) * ((1 - S) ** 6))
    Bco[5] = 5 * (300 + 1579 * S + 3176 * (S ** 2) + 2949 * (S ** 3) + 1188 * (S ** 4) + 675 * (S ** 5)
                  + 1326 * (S ** 6) + 827 * (S ** 7) + 130 * (S ** 8)) / (384 * (3 + 2 * S) * (4 + S) * ((1 - S) ** 6))
    # Calculation of the C coefficients
    Cco = np.empty(3)
    Cco[0] = np.sqrt(np.tanh(kd))
    Cco[1] = (np.sqrt(np.tanh(kd)) * (2 + 7 * S ** 2)) / (4 * (1-S) ** 2)
    Cco[2] = (np.sqrt(np.tanh(kd)) * (4 + 32 * S - 116 * S ** 2 - 400 * S ** 3 - 71 * S ** 4 + 146 * S ** 5)) \
        / (32 * (1 - S) ** 5)
    # Calculation of the D coefficients
    Dco = np.empty(2)
    Dco[0] = -0.5 * np.sqrt(1 / np.tanh(kd))
    Dco[1] = (np.sqrt(1 / np.tanh(kd)) * (2 + 4 * S + S ** 2 + 2 * S ** 3)) / (8 * (1 - S) ** 3)
    # Calculation of the E coefficients
    Eco = np.empty(2)
    Eco[0] = (np.tanh(kd) * (2 + 2 * S + 5 * S ** 2)) / (4 * (1 - S) ** 2)
    Eco[1] = (np.tanh(kd) * (8 + 12 * S - 152 * S ** 2 - 308 * S ** 3 - 42 * S ** 4 + 77 * S ** 5)) \
        / (32 * (1 - S) ** 5)

    # calculate properties
    # Initialising coefficients
    A11 = Aco[0]
    A22 = Aco[1]
    A31 = Aco[2]
    A33 = Aco[3]
    A42 = Aco[4]
    A44 = Aco[5]
    A51 = Aco[6]
    A53 = Aco[7]
    A55 = Aco[8]
    B22 = Bco[0]
    B31 = Bco[1]
    B42 = Bco[2]
    B44 = Bco[3]
    B53 = Bco[4]
    B55 = Bco[5]
    C0 = Cco[0]
    
    # Wave steepness
    epsilon = A * k #TODO
    # epsilon = A * (kx + ky)
    #
    
    psi = -(omega * t - kx * xpoints - ky * ypoints)
    
    eta = (1 / k) * (epsilon * np.cos(psi) + B22 * (epsilon ** 2) * np.cos(2 * psi)
                     + B31 * (epsilon ** 3) * (np.cos(psi) - np.cos(3 * psi))
                     + (epsilon ** 4) * (B42 * np.cos(2 * psi) + B44 * np.cos(4 * psi))
                     + (epsilon ** 5) * (-(B53 + B55) * np.cos(psi) + B53 * np.cos(3 * psi)
                     + B55 * np.cos(5 * psi)))

    eta.shape = (numx, numy)                  

    return eta


    
if __name__ == '__main__':

    
    h = 100 # depth
    T = 20 # period
    H = 35 # wave height
    
    k, omega = sk1.fDispersionSTOKES5(h, H, T)

    A = H / 2
    theta = np.pi/4

    numx = 30
    numy = 30

    xrange = np.linspace(-500, 500, numx)
    yrange = np.linspace(-500, 500, numy)
    X, Y = np.meshgrid(xrange, yrange)

    nt = 100
    trange = np.linspace(0,100,nt)
    names = []

    for t in trange:
        eta = stokes_wave_surface(k, h, A, xrange, yrange, omega, t, theta)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        surf = ax.plot_surface(X, Y, eta)

        name = f'time_{t}.png'
        names.append(name)

        plt.savefig(name)
        plt.close()

    with imageio.get_writer('stokesmoving.gif', mode='I') as writer:
        for filename in names:
            image = imageio.imread(filename)
            writer.append_data(image)

    for name in set(names):
        os.remove(name)