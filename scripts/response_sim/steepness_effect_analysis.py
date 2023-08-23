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
    freq = 4.00
    period = 300  # total time range
    max_u = np.empty((z_num, len(q_stps)))
    max_du = np.empty((z_num, len(q_stps)))

    for i_s, stp in enumerate(q_stps):

        tp = np.sqrt((hs*2*np.pi)/(stp*9.81))

        ss = spctr.SeaState(hs=np.array([hs]), tp=np.array([10]), spctr_type=spctr.Jonswap)

        np.random.seed(1)

        lin_wave = kin.LinearKin(sample_f=4.00, period=100, z_values=z_range, sea_state=ss)
        lin_wave.compute_spectrum()
        lin_wave.compute_kinematics(cond=False)
        _, u, _, du, _ = lin_wave.retrieve_kinematics()
        max_u[:, i_s] = np.max(np.abs(u[:, :, 0]), axis=0)
        max_du[:, i_s] = np.max(np.abs(du[:, :, 0]), axis=0)

    plt.subplot(2, 1, 1)
    plt.plot(max_u, z_range+depth)
    plt.xlabel("u [m/s]")
    plt.ylabel("depth [m] (above sea bed)")

    plt.subplot(2, 1, 2)
    plt.plot(max_du, z_range+depth)
    plt.xlabel("du [m/s]")
    plt.ylabel("depth [m] (above sea bed)")

    plt.show()
