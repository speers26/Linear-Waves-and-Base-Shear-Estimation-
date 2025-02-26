import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load csvs
env_probs = pd.read_csv('/home/speersm/GitHub/Linear-Waves-and-Base-Shear-Estimation-/2024Paper_scripts/env_probs.csv')
cde = pd.read_csv('/home/speersm/GitHub/Linear-Waves-and-Base-Shear-Estimation-/2024Paper_scripts/cond_dens_duo.csv', header=None)

# plot heatmap of cde over env_probs x and y
x = np.array(sorted(set(env_probs['x'])))
y = np.array(sorted(set(env_probs['y'])))
cde = np.array(cde).reshape(len(y), len(x))

X, Y = np.meshgrid(x, y)
plt.contourf(X, Y, cde)
plt.xlabel('Hs [m]')
plt.ylabel('Tp [s]')
plt.colorbar()
plt.show()