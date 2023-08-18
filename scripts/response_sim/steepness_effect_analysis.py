import numpy as np
import pandas as pd
import wavesim.spectrum as spctr
import wavesim.kinematics as kin
import matplotlib.pyplot as plt

if __name__ == "__main__":

    stp_range = np.arange(start=0.01, stop=0.08)
    hs = 15

    for stp in stp_range:

        tp = np.sqrt((hs*2*np.pi)/(stp*9.81))

        ss = spctr.SeaState(hs=np.array([hs]), tp=np.array([10]), spctr_type=spctr.Jonswap)

        np.random.seed(1)

        depth = 100
        z_num = 150
        z_range = np.linspace(-depth, 50, z_num)
        freq = 4.00
        period = 300  # total time range

        lin_wave = kin.LinearKin(sample_f=4.00, period=100, z_values=z_range, sea_state=ss)
        lin_wave.compute_spectrum()
        lin_wave.compute_kinematics(cond=False)
        lin_wave.plot_kinematics()
