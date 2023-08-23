import numpy as np
import pandas as pd
import wavesim.spectrum as spctr
import wavesim.kinematics as kin
import matplotlib.pyplot as plt

if __name__ == "__main__":

    q_stps = np.linspace(start=0.01, stop=0.08, num=50)
    hs = 15

    depth = 100
    z_num = 151
    z_range = np.linspace(-depth, 50, z_num)
    q_zs = z_range
    freq = 4.00
    period = 300  # total time range
    max_u = np.empty((len(q_stps), len(q_zs), 1))
    max_du = np.empty((len(q_stps), len(q_zs), 1))

    for i_s, stp in enumerate(q_stps):

        tp = np.sqrt((hs*2*np.pi)/(stp*9.81))

        ss = spctr.SeaState(hs=np.array([hs]), tp=np.array([10]), spctr_type=spctr.Jonswap)

        np.random.seed(1)

        lin_wave = kin.LinearKin(sample_f=4.00, period=100, z_values=z_range, sea_state=ss)
        lin_wave.compute_spectrum()
        lin_wave.compute_kinematics(cond=False)
        _, u, _, du, _ = lin_wave.retrieve_kinematics()
        max_u[i_s, :, 0] = np.abs(np.max(u[:, np.in1d(z_range, q_zs), 0], axis=0))
        max_du[i_s, :, 0] = np.abs(np.max(du[:, np.in1d(z_range, q_zs), 0], axis=0))

    query_z = 10.0
    plt.subplot(2, 1, 1)
    plt.plot(q_stps, max_u[:, np.where(z_range == query_z)[0], 0].reshape(50,))
    plt.subplot(2, 1, 2)
    plt.plot(q_stps, max_du[:, np.where(z_range == query_z)[0], 0].reshape(50,))
    plt.show()
