import numpy as np
import matplotlib.pyplot as plt
from wavesim.kinematics import stokes_kinematics
from wavesim.loading import morison_load
from wavesim.dispersion import fDispersionSTOKES5
from wavesim import kinematics as kin

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement

    h = 100  # depth
    T = 20  # period
    H = 35  # wave height

    n_depth = 151
    z_range = np.linspace(-h, 50, n_depth)

    n_time = 200
    time = np.linspace(-20, 20, 200)

    stokes_wave = kin.StokesKin(t_values=time, z_values=z_range, H = H, T=T)

    stokes_wave.compute_kinematics()
    stokes_wave.plot_kinematics()

    stokes_wave.compute_base_shear()
    stokes_wave.plot_base_shear()
