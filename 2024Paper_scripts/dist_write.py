import numpy as np
import pandas as pd
import wavesim.distest as dist
import wavesim.spectrum as spctr
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    env_probs = pd.read_csv('scripts/response_sim/env_probs.csv')
    env_probs = env_probs[env_probs.p != 0].reset_index()

    num_sea_states = 2000
    z_values = np.linspace(-100, 50, 150)

    x_num = 1000
    X = np.linspace(0, 20, num=x_num)

    # pick cd, cdms
    cm_l = 1.0
    cm_u = 100.0
    cd_l = 1.0
    cd_u = 100.0
    deck_height = 20.0

    diffs = abs(z_values-deck_height)
    deck_ind = np.where(diffs == np.min(diffs))[0][0]
    c_m = np.concatenate((np.tile(cm_l, deck_ind), np.tile(cm_u, len(z_values)-deck_ind)))
    c_d = np.concatenate((np.tile(cd_l, deck_ind), np.tile(cd_u, len(z_values)-deck_ind)))

    np.random.seed(1)

    print(env_probs.shape[0])
    cond_dists = []

    for s in range(env_probs.shape[0]):
        print(s)
        row = env_probs.loc[s]
        hs = np.tile(row['x'], num_sea_states)
        s2 = np.tile(row['y'], num_sea_states)
        tp = np.sqrt((hs*2*np.pi)/(s2*9.81))
        ss = spctr.SeaState(hs=hs, tp=tp, spctr_type=spctr.Jonswap)

        loadEst = dist.MorisonDistEst(sea_state=ss, z_values=z_values, c_d=c_d, c_m=c_m)
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
        cond_dists.append(loadEst)

    # pickle dump
    save_object(cond_dists, 'scripts/cond_dists.pkl')
