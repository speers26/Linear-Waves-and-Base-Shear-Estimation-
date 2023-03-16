import numpy as np
from wavesim import kinematics as kin
from wavesim import loading as load

if __name__ == '__main__':

    h = 100  # depth
    T = 20  # period
    H = 35  # wave height

    n_depth = 151
    z_range = np.linspace(-h, 50, n_depth)

    n_time = 200
    time = np.linspace(-20, 20, n_time)

    stokes_wave = kin.StokesKin(t_values=time, z_values=z_range, H=H, T=T)
    stokes_wave.compute_kinematics()
    stokes_wave.plot_kinematics()

    stokes_load = load.MorisonLoad(stokes_wave)
    stokes_load.compute_load()
    stokes_load.plot_load()
