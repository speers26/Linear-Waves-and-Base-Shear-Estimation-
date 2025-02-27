import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp

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


def evaluate_stored_cdf(s: list, X: np.ndarray):
    """for parallelisation

    Args:
        s (list): index of cond_dists.pkl
        X (np.ndarray): evaluation points

    Returns:
        np.ndarray: cdf evaluated at given points
    """

    print(s)
    cdf_array = cond_dists[s].eval_cdf(X, smooth=False)

    return cdf_array


def evaluate_stored_pdf(s: list, X: np.ndarray):
    """for parallelisation

    Args:
        s (list): index of cond_dists.pkl
        X (np.ndarray): evaluation points

    Returns:
        np.ndarray: pdf evaluated at given points
    """

    print(s)
    pdf_array = cond_dists[s].eval_pdf(X, smooth=False)

    return pdf_array


if __name__ == "__main__":

    # average number of storms per year
    lamda = 73

    # return period
    period = 1000  # years

    # reading in conditioned distribution estimates
    with open('2024Paper_scripts/cond_dists_duo_200_100.pkl', 'rb') as inp:
        cond_dists = pickle.load(inp)
    cond_dists = [c[1] for c in cond_dists]

    # reading in environment density
    env_probs = pd.read_csv('2024Paper_scripts/env_probs.csv')
    nfull = len(env_probs['p'])
    env_probs = env_probs[env_probs.p != 0]

    # getting marginal 3 hour response distribution ----------------------------------------
    X = np.linspace(0, 300, num=1000)  # this needs to be selected carefully

    cl = mp.Pool(4)
    cdf_list = cl.starmap(evaluate_stored_cdf, [[i, X] for i in range(env_probs.shape[0])])
    cl.close()

    cdf_array = np.empty((env_probs.shape[0], len(X)))
    for i in range(len(cond_dists)):
        cdf_array[i, :] = cdf_list[i]

    p_array = np.array(env_probs['p'])
    f_cdf = np.sum(cdf_array * p_array[:, np.newaxis], axis=0)

    # getting marginal annual response distribution ----------------------------------------
    cdf_an = np.exp(-lamda*(1-f_cdf))

    # getting return value
    rp = np.round(return_level(period, cdf_an, X), 3)

    # getting conditional density ----------------------------------------------------------
    cl = mp.Pool(4)
    rp_cond_theta = cl.starmap(evaluate_stored_pdf, [[i, np.array([rp])] for i in range(env_probs.shape[0])])
    cl.close()

    # rp_marg = np.tile(eval_pdf(rp, mids, f_pdf), len(cond_dists))
    rp_cond_theta = np.concatenate(rp_cond_theta, axis=0)
    dens_quotient = (rp_cond_theta / np.sum(rp_cond_theta * env_probs['p']))
    f_theta_r = np.array(env_probs['dens']) * dens_quotient
    p_theta_r = np.array(env_probs['p']) * dens_quotient

    f_theta_r_w0 = np.tile(0.0, nfull)
    f_theta_r_w0[env_probs.index] = f_theta_r

    # write to file
    np.savetxt('2024Paper_scripts/cond_dens_duo_200_100.csv', f_theta_r_w0, delimiter=',')

    # failure prob region ------------------------------------------------------------------
    rc = rp

    cl = mp.Pool(4)
    fail_ps = cl.starmap(evaluate_stored_cdf, [[i, np.array([rp])] for i in range(env_probs.shape[0])])
    fail_ps = [[1-i] for i in fail_ps]
    fail_ps = np.array(fail_ps).reshape(len(env_probs['dens']),)
    cl.close()
    fail_ps_full = np.tile(0.0, nfull)
    fail_ps_full[env_probs.index] = fail_ps

    # write to file
    np.savetxt('2024Paper_scripts/fail_ps_duo_200_100.csv', fail_ps_full, delimiter='')
