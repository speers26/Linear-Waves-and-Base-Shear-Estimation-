import numpy as np
import pandas as pd
import wavesim.distest as dist
import wavesim.spectrum as spctr
import pickle


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    env_probs = pd.read_csv('scripts/env_probs.csv')
    env_probs = env_probs[env_probs.p != 0].reset_index()

    num_sea_states = 2000
    z_values = np.linspace(-100, 50, 50)

    x_num = 1000
    X = np.linspace(0, 20, num=x_num)

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

        loadEst = dist.LoadDistEst(sea_state=ss, z_values=z_values)
        loadEst.compute_cond_crests()
        loadEst.compute_kinematics()
        loadEst.compute_load()
        loadEst.compute_sea_state_max()
        loadEst.compute_is_distribution(X=X)
        loadEst.compute_density()

        # save to vector
        cond_dists.append(loadEst)

    # pickle dump
    save_object(cond_dists, 'scripts/cond_dists.pkl')
