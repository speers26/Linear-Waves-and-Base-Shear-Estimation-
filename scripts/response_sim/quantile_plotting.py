import pickle
import pandas as pd
import cond_dens_est as cde
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    with open('scripts/response_sim/cond_dists.pkl', 'rb') as f:
        cond_dists = pickle.load(f)
    cond_dists = [c[1] for c in cond_dists]

    env_probs = pd.read_csv('scripts/response_sim/env_probs.csv')

    if len(cond_dists) != env_probs.shape[0]:
        raise Exception("Conditioned distributions do not match environment density")

    q = 0.5
    period = 1/(1-q)
    qs = np.empty(len(cond_dists))

    for s in range(len(cond_dists)):

        plt.plot(cond_dists[s].X, cond_dists[s].cdf)
        plt.show()
        qs[s] = cde.return_level(period, cond_dists[s].cdf, cond_dists[s].X)

    np.savetxt('/home/speersm/GitHub/environment-modelling/data/test_quantiles.csv', qs, delimiter='')
