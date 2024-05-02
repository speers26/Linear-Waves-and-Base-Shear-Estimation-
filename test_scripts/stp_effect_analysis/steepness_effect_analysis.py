import numpy as np
import pandas as pd
import wavesim.spectrum as spctr
import wavesim.kinematics as kin
import matplotlib.pyplot as plt

if __name__ == "__main__":

    q_stps = np.linspace(start=0.01, stop=0.08, num=3)
    hs = 15
    a = 30

    depth = 100
    z_num = 50
    z_range = np.linspace(-depth, 50, z_num)
    freq = 4.00
    period = 300  # total time range
    mid_u = np.empty((z_num, len(q_stps)))
    mid_du = np.empty((z_num, len(q_stps)))

    for i_s, stp in enumerate(q_stps):

        tp = np.sqrt((hs*2*np.pi)/(stp*9.81))

        ss = spctr.SeaState(hs=np.array([hs]), tp=np.array([tp]), spctr_type=spctr.Jonswap)

        np.random.seed(1)

        lin_wave = kin.LinearKin(sample_f=4.00, period=100, z_values=z_range, sea_state=ss)
        lin_wave.compute_spectrum()
        lin_wave.compute_kinematics(cond=True, a=[a])
        _, u, _, du, _ = lin_wave.retrieve_kinematics()
        mid_u[:, i_s] = u[int(lin_wave.nt/2), :, 0]
        mid_du[:, i_s] = du[int(lin_wave.nt/2), :, 0]

    plt.subplot(2, 1, 1)
    plt.axhline(y=depth+a, color='r', linestyle='--')
    plt.axhline(y=depth, color='r', linestyle='dashdot')
    plt.plot(mid_u, z_range+depth)
    plt.xlabel("u [m/s]")
    plt.ylabel("depth [m] (above sea bed)")

    plt.subplot(2, 1, 2)
    plt.axhline(y=depth+a, color='r', linestyle='--')
    plt.axhline(y=depth, color='r', linestyle='dashdot')
    plt.plot(mid_du, z_range+depth)
    plt.xlabel("du [m/s]")
    plt.ylabel("depth [m] (above sea bed)")

    plt.show()
