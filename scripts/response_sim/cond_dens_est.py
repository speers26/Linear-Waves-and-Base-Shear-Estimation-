import pickle
import numpy as np
import pandas as pd


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


def eval_pdf(x: np.ndarray, mids: np.ndarray, pdf: np.ndarray):
    """evaulates the gived pdf at a point x by assigning the
    density at the closest defined point

    Args:
        x (np.ndarray): point to evaluate at
        mids (np.ndarray): points pdf is defined at
        pdf (np.ndarray): densities defined at mids

    Returns:
        float: density
    """
    abs_diffs = np.abs(mids-x)
    close_mid_ind = np.where(abs_diffs == np.min(abs_diffs))
    return pdf[close_mid_ind]


if __name__ == "__main__":

    # average number of storms per year
    lamda = 100

    # return period
    period = 1000  # years

    # reading in conditioned distribution estimates
    with open('scripts/response_sim/cond_dists.pkl', 'rb') as inp:
        cond_dists = pickle.load(inp)
    cond_dists = [c[1] for c in cond_dists]

    # reading in environment density
    env_probs = pd.read_csv('scripts/response_sim/env_probs.csv')
    nfull = len(env_probs['p'])

    # getting marginal 3 hour response distribution
    X = np.linspace(-5, 1000, num=1000)
    cdf_array = np.empty((env_probs.shape[0], len(X)))
    for i in range(len(cond_dists)):
        print(i)
        cdf_array[i, :] = cond_dists[i].eval_cdf(X, smooth=True)
    p_array = np.array(env_probs['p'])
    f_cdf = np.sum(cdf_array * p_array[:, np.newaxis], axis=0)
    # x_num = len(f_cdf)
    # X = np.linspace(min(cond_dists[0].X), max(cond_dists[0].X), num=x_num)

    # getting marginal annual response distribution
    cdf_an = np.exp(-lamda*(1-f_cdf))

    # getting return value
    rp = np.round(return_level(period, cdf_an, X), 3)

    # # getting 3 hour max density
    # dx = X[1] - X[0]
    # mids = (X[1:] + X[:-1]) / 2
    # f_pdf = np.diff(f_cdf)/dx
    # f_pdf = f_pdf/(np.sum(f_pdf * dx))

    # getting conditioned density at rp (unormalised)
    rp_cond_theta = np.empty(len(cond_dists))
    for i in range(len(cond_dists)):
        rp_cond_theta[i] = cond_dists[i].eval_pdf(np.array([rp]), smooth=True)

    # rp_marg = np.tile(eval_pdf(rp, mids, f_pdf), len(cond_dists))
    dens_quotient = rp_cond_theta / np.sum(rp_cond_theta)
    f_theta_r = np.array(env_probs['dens']) * dens_quotient
    p_theta_r = np.array(env_probs['p']) * dens_quotient

    f_theta_r_w0 = np.tile(0.0, nfull)
    f_theta_r_w0[env_probs.index] = f_theta_r

    # write to files
    np.savetxt('/home/speersm/GitHub/environment-modelling/data/cond_dens.csv', f_theta_r_w0,
               delimiter='')
    np.savetxt('cond_dens.csv', f_theta_r_w0, delimiter=',')

    # failure prob region
    rc = rp

    fail_ps = np.empty(len(cond_dists))
    for i_c, c in enumerate(cond_dists):
        print(i_c)
        fail_ps[i_c] = 1-cond_dists[i_c].eval_cdf(np.array([rc]), smooth=False)
        # eval_pdf(rc, X, cond_dists[i_c].cdf)

    fail_ps_full = np.tile(0.0, nfull)
    fail_ps_full[env_probs.index] = fail_ps

    np.savetxt('/home/speersm/GitHub/environment-modelling/data/fail_ps.csv', fail_ps_full, delimiter='')
