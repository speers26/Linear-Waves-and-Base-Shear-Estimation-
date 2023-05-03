import numpy as np
from wavesim import kinematics as kin
from wavesim import loading as load
from wavesim import spectrum as spctr

if __name__ == '__main__':

    H = np.array([15])
    T = np.array([10])
    ss1 = spctr.SeaState(H_det=H, T_det=T)

    z_num = 150
    z_range = np.linspace(-100, 50, z_num)
    freq = 1.00  # 3. / (2*np.pi)
    period = 100  # total time range

    stokes_wave = kin.StokesKin(freq, period, z_range, ss1)
    stokes_wave.compute_kinematics()
    stokes_wave.plot_kinematics()

    stokes_load = load.MorisonLoad(stokes_wave)
    stokes_load.compute_load()
    stokes_load.plot_load()
