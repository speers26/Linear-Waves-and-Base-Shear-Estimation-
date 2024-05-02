import numpy as np
import wavesim.spectrum as spctr
import wavesim.kinematics as kin
import matplotlib.pyplot as plt
import wavesim.loading as load

if __name__ == "__main__":

    q_stps = np.linspace(start=0.01, stop=0.08, num=3)
    hs = 15

    depth = 100
    z_num = 50
    z_range = np.linspace(-depth, 50, z_num)
    freq = 4.00
    period = 100  # total time range

    num_sea_states = 1

    # pick cd, cdms
    cm_l = 1.0
    cm_u = 1.0
    cd_l = 1.0
    cd_u = 1.0
    deck_height = 5.0

    a = 30
    cond = True

    diffs = abs(z_range-deck_height)
    deck_ind = np.where(diffs == np.min(diffs))[0][0]
    c_m = np.concatenate((np.tile(cm_l, deck_ind), np.tile(cm_u, len(z_range)-deck_ind)))
    c_d = np.concatenate((np.tile(cd_l, deck_ind), np.tile(cd_u, len(z_range)-deck_ind)))

    for i_s, s in enumerate(q_stps):

        print(i_s)
        s2 = s

        np.random.seed(1)

        tp = np.sqrt((hs*2*np.pi)/(s2*9.81))
        ss = spctr.SeaState(hs=np.tile(hs, num_sea_states), tp=np.tile(tp, num_sea_states), spctr_type=spctr.Jonswap)

        lin_wave = kin.LinearKin(sample_f=freq, period=period, z_values=z_range, sea_state=ss)
        lin_wave.compute_spectrum()
        lin_wave.compute_kinematics(cond=cond, a=[a])

        lin_load = load.MorisonLoad(lin_wave, c_d, c_m)
        lin_load.compute_load()
        load_ts = lin_load.retrieve_load()
        eta, _, _, _, _ = lin_wave.retrieve_kinematics()

        plt.subplot(2, 1, 1)
        plt.plot(lin_wave.t_values, eta)

        plt.subplot(2, 1, 2)
        plt.plot(lin_wave.t_values, load_ts)


plt.show()
