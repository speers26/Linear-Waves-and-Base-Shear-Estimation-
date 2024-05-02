import multiprocessing as mp
import numpy as np
import wavesim.spectrum as spctr
import wavesim.kinematics as kin
import wavesim.loading as loading
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from numpy import diff


def compute_max_response(a: np.ndarray):
    """computes the maximum response for given crest heights

    Args:
        a (np.ndarray): crest heights
    """

    lin_wave = kin.LinearKin(sample_f=4.00, period=100, z_values=z_range, sea_state=ss)
    lin_wave.compute_spectrum()
    lin_wave.compute_kinematics(cond=True, a=a)

    lin_load = loading.MorisonLoad(lin_wave, c_d, c_m)

    lin_load.compute_load()
    max_series = np.empty(len(a))

    for s in range(len(a)):
        load = lin_load.retrieve_load()
        load = load[:, s]
        # get maximums
        mins = argrelextrema(load, np.less)[0]
        lower_min = np.max(mins[mins < lin_wave.nt/2])
        upper_min = np.min(mins[mins > lin_wave.nt/2])
        slice = load[lower_min:upper_min]
        max_series[s] = max(slice)

    return max_series


if __name__ == "__main__":

    a = np.array([10, 15, 20, 25, 30])

    np.random.seed(1)

    depth = 100
    ss = spctr.SeaState(hs=np.tile(15, len(a)), tp=np.tile(10, len(a)), spctr_type=spctr.Jonswap)

    z_num = 150
    z_range = np.linspace(-depth, 50, z_num)
    freq = 4.00
    period = 100  # total time range

    # pick cd, cdms
    cm_l = 1.0
    cm_u = 100.0
    cd_l = 1.0
    cd_u = 100.0
    deck_height = 20.0

    diffs = abs(z_range-deck_height)
    deck_ind = np.where(diffs == np.min(diffs))[0][0]
    c_m = np.concatenate((np.tile(cm_l, deck_ind), np.tile(cm_u, 2), np.tile(cm_l, z_num-deck_ind-2)))
    c_d = np.concatenate((np.tile(cd_l, deck_ind), np.tile(cd_u, 2), np.tile(cd_l, z_num-deck_ind-2)))

    plt.subplot(1, 2, 1)
    plt.plot(z_range, c_m)
    plt.title("c_m")

    plt.subplot(1, 2, 2)
    plt.plot(z_range, c_d)
    plt.title("c_d")
    plt.show()

    loads = compute_max_response(a)

    plt.plot(a, loads)
    plt.xlabel("crest height [m]")
    plt.ylabel("reponse [MN]")
    plt.show()
