import pickle
import pandas as pd
import cond_dens_est as cde
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp


def evaluate_stored_cdf(s: list, X: np.ndarray):
    """for parallelisation

    Args:
        s (list): index of cond_dists.pkl
        X (np.ndarray): evaluation points

    Returns:
        np.ndarray: cdf evaluated at given points
    """

    print(s)
    cdf_array_0 = cond_dists[s].eval_cdf(X, smooth=False)
    X_max = max(X[np.where(cdf_array_0 < 1.0)])
    X_array = np.linspace(0, X_max, 2000)
    cdf_array = cond_dists[s].eval_cdf(X_array, smooth=False)
    return X_array, cdf_array


def return_level(period: int, cdf: np.ndarray, X: np.ndarray):
    """gives return level for given period and numeric cdf

    Args:
        period (int): desired return period
        cdf (np.ndarray): emprical cdf
        X (np.ndarray): x points for cdf

    Returns:
        float: return value
    """
    p = 1 - 1 / period
    diffs = np.abs(p - cdf)
    rp = X[np.where(diffs == np.min(diffs))]
    return float(rp)


if __name__ == "__main__":

    with open('scripts/response_sim/cond_dists.pkl', 'rb') as f:
        cond_dists = pickle.load(f)
    cond_dists = [c[1] for c in cond_dists]

    # reading in environment density
    env_probs = pd.read_csv('scripts/response_sim/env_probs.csv')

    # getting marginal 3 hour response distribution ----------------------------------------
    X = np.linspace(-5, 300, num=1000)

    cl = mp.Pool(4)
    cdf_list = cl.starmap(evaluate_stored_cdf, [[i, X] for i in range(env_probs.shape[0])])
    cl.close()

    q = 0.5
    period = 1/(1-q)
    qs = np.empty(len(cond_dists))

    for s in range(len(cond_dists)):

        plt.plot(X, cdf_list[s])
        plt.show()
        qs[s] = return_level(period, cdf_list[s], X)

    np.savetxt('/home/speersm/GitHub/environment-modelling/data/test_quantiles.csv', qs, delimiter='')
