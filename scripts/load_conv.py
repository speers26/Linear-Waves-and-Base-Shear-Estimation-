import numpy as np
import pandas as pd
from wavesim.loading import LoadDistEst
import matplotlib.pyplot as plt

env_probs = pd.read_csv('scripts/env_probs.csv')

num_sea_states = 2000
sea_state_hours = 3
z_values = np.linspace(-100, 50, 50)

x_num = 100
X = np.linspace(0, 10, num=x_num)

np.random.seed(1)

results = np.empty((env_probs.shape[0], x_num))

for i_s in range(env_probs.shape[0]):
    print(i_s)
    row = env_probs.loc[i_s]
    hs = row['hs']
    s2 = row['s2']
    tp = np.sqrt((hs*2*np.pi)/(s2*9.81))

    loadEst = LoadDistEst(hs=hs, tp=tp, num_sea_states=num_sea_states, sea_state_hours=sea_state_hours, z_values=z_values)

    loadEst.compute_tf_values()
    loadEst.compute_spectrum()
    loadEst.compute_cond_crests()

    loadEst.simulate_sea_states()
    loadEst.compute_load_dist()

    results[i_s, :] = loadEst.load_cdf

np.savetxt("scripts/results.csv", results, delimiter=",")
p_array = np.array(env_probs['p'])
f_cdf = np.sum(results * p_array[:, np.newaxis], axis=0)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(X, f_cdf)
plt.subplot(2, 1, 2)
plt.plot(X, np.log10(1-f_cdf))
plt.show()
