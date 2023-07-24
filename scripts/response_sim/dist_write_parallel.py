import multiprocessing as mp
import numpy as np
import pandas as pd
import wavesim.distest as dist
import wavesim.spectrum as spctr
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def compute_response_dist(s: list):

    print(s)
    row = env_probs.loc[s]
    hs = np.tile(row['x'], num_sea_states)
    s2 = np.tile(row['y'], num_sea_states)
    tp = np.sqrt((hs*2*np.pi)/(s2*9.81))
    ss = spctr.SeaState(hs=hs, tp=tp, spctr_type=spctr.Jonswap)

    loadEst = dist.LoadDistEst(sea_state=ss, z_values=z_values)
    loadEst.compute_cond_crests()
    loadEst.compute_kinematics()
    loadEst.compute_load()
    loadEst.compute_sea_state_max()
    loadEst.compute_is_distribution(X=X)
    loadEst.compute_density()
    loadEst.kinematics = 0
    loadEst.max_series = 0
    loadEst.load = 0

    # save to vector
    return s, loadEst


if __name__ == '__main__':

    env_probs = pd.read_csv('scripts/response_sim/env_probs.csv')
    env_probs = env_probs[env_probs.p != 0].reset_index()

    num_sea_states = 2000
    z_values = np.linspace(-100, 50, 50)

    x_num = 1000
    X = np.linspace(0, 20, num=x_num)

    np.random.seed(1)

    print(env_probs.shape[0])
    cond_dists = []

    cl = mp.Pool(4)
    index_cond_dists = cl.map(compute_response_dist, [i for i in range(env_probs.shape[0])])

    # pickle dump

    save_object(index_cond_dists, 'scripts/response_sim/cond_dists.pkl')