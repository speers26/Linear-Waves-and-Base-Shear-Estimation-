import numpy as np
from wavesim import kinematics as kin

if __name__ == '__main__':

    h = 100
    T = 20
    A = 20/2

    n_depth = 151
    z_range = np.linspace(-h, 50, n_depth)

    n_time = 200
    time = np.linspace(-20, 20, n_time)

    airy_wave = kin.AiryKin(t_values=time, z_values=z_range, H=A*2, T=T)
    airy_wave.compute_kinematics()
    airy_wave.plot_kinematics()

    airy_wave.compute_base_shear()
    airy_wave.plot_base_shear()
