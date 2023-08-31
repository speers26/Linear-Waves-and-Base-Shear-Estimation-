import numpy as np
import pandas as pd
import wavesim.spectrum as spctr
import wavesim.kinematics as kin
import matplotlib.pyplot as plt
import wavesim.distest as dist

if __name__ == "__main__":

    q_stps = np.linspace(start=0.01, stop=0.08, num=3)
    hs = 30

    depth = 100
    z_num = 50
    z_values = np.linspace(-depth, 50, z_num)
    freq = 4.00
    period = 300  # total time range

    num_sea_states = 2000
    x_num = 1000
    X = np.linspace(0, 1000, num=x_num)

    # pick cd, cdms
    cm_l = 1.0
    cm_u = 100.0
    cd_l = 1.0
    cd_u = 100.0
    deck_height = 5.0

    diffs = abs(z_values-deck_height)
    deck_ind = np.where(diffs == np.min(diffs))[0][0]
    c_m = np.concatenate((np.tile(cm_l, deck_ind), np.tile(cm_u, len(z_values)-deck_ind)))
    c_d = np.concatenate((np.tile(cd_l, deck_ind), np.tile(cd_u, len(z_values)-deck_ind)))

    for i_s, s in enumerate(q_stps):

        print(i_s)
        s2 = s

        tp = np.sqrt((hs*2*np.pi)/(s2*9.81))
        ss = spctr.SeaState(hs=np.tile(hs, num_sea_states), tp=np.tile(tp, num_sea_states), spctr_type=spctr.Jonswap)

        loadEst = dist.MorisonDistEst(sea_state=ss, z_values=z_values, c_d=c_d, c_m=c_m)
        loadEst.compute_cond_crests()
        loadEst.compute_kinematics()
        loadEst.compute_load()
        loadEst.compute_sea_state_max()
        loadEst.compute_is_distribution(X=X)
        plt.plot(X, loadEst.cdf)

plt.show()
