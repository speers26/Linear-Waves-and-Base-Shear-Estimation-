import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# structure C cdcm
cdcm = 10000

# load csvs
env_probs = pd.read_csv('/home/speersm/GitHub/Linear-Waves-and-Base-Shear-Estimation-/2024Paper_scripts/env_probs.csv')
cde = pd.read_csv(f'/home/speersm/GitHub/Linear-Waves-and-Base-Shear-Estimation-/2024Paper_scripts/cond_dens_duo_{cdcm}_100.csv', header=None)

# plot heatmap of cde over env_probs x and y
cde = np.array(cde)
x = np.array(env_probs['x'])
y = np.array(env_probs['y'])

plt.scatter(x, y, c=cde, cmap='hot_r', marker='s')
plt.colorbar(label='CDE')
plt.xlabel('Hs [m]')
plt.ylabel('Tp [s]')
plt.show()