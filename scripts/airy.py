import numpy as np
from wavesim import kinematics as kin
from wavesim import loading as load
from wavesim import spectrum as spctr

if __name__ == '__main__':

    H = np.array([10, 15])
    T = np.array([10, 12])
    ss1 = spctr.SeaState(H_det=H, T_det=T)

    z_num = 150
    z_range = np.linspace(-100, 50, z_num)
    freq = 4.00  # 3. / (2*np.pi)
    period = 60  # total time range

    airy_wave = kin.AiryKin(sample_f=freq, period=period, z_values=z_range, sea_state=ss1)
    airy_wave.compute_kinematics()
    airy_wave.plot_kinematics(s=0)

    airy_load = load.MorisonLoad(airy_wave)
    airy_load.compute_load()
    airy_load.plot_load()
