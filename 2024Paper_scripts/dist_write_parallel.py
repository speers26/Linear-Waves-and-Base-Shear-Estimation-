import multiprocessing as mp
import numpy as np
import pandas as pd
import wavesim.distest as dist
import wavesim.spectrum as spctr
import pickle
import matplotlib.pyplot as plt


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

    loadEst = dist.MorisonDuoDistEst(sea_state=ss, z_values=z_values, c_d_1=c_d_1, c_m_1=c_m_1, c_d_2=c_d_2, c_m_2=c_m_2)
    loadEst.compute_cond_crests()
    loadEst.compute_kinematics()
    loadEst.compute_load()
    loadEst.compute_sea_state_max()
    loadEst.compute_pdf()
    loadEst.compute_cdf()

    loadEst.max_series = 0

    # save to vector
    return s, loadEst


if __name__ == '__main__':

    env_probs = pd.read_csv('env_probs.csv')
    env_probs = env_probs[env_probs.p != 0].reset_index()

    num_sea_states = 2000
    z_values = np.linspace(-100, 50, 50)

    x_num = 1000
    X = np.linspace(0, 100, num=x_num)

    # pick cd, cdms
    # set all to 1.0 for structure A
    cm_l = 1.0
    cm_u = 100.0
    cd_l = 1.0
    cd_u = 100.0
    deck_height = -5.0  # change to 5 for structure B

    diffs = abs(z_values-deck_height)
    deck_ind = np.where(diffs == np.min(diffs))[0][0]

    c_m_1 = np.concatenate((np.tile(cm_l, deck_ind), np.tile(cm_u, 3), np.tile(cm_l, len(z_values)-deck_ind-3)))
    c_d_1 = np.concatenate((np.tile(cd_l, deck_ind), np.tile(cd_u, 3), np.tile(cd_l, len(z_values)-deck_ind-3)))

    # set all to 1.0 for structure A
    cm_l = 1.0
    cm_u = 100.0
    cd_l = 1.0
    cd_u = 100.0
    deck_height = 5.0  # change to -5 for structure C

    c_m_2 = np.concatenate((np.tile(cm_l, deck_ind), np.tile(cm_u, 3), np.tile(cm_l, len(z_values)-deck_ind-3)))
    c_d_2 = np.concatenate((np.tile(cd_l, deck_ind), np.tile(cd_u, 3), np.tile(cd_l, len(z_values)-deck_ind-3)))

    np.random.seed(1)

    print(env_probs.shape[0])

    cl = mp.Pool(89)
    cond_dists = cl.map(compute_response_dist, [i for i in range(env_probs.shape[0])])
    cl.close()

    # pickle dump
    save_object(cond_dists, 'cond_dists_duo.pkl')
