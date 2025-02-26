import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load csvs
env_probs = pd.read_csv('/home/speersm/GitHub/Linear-Waves-and-Base-Shear-Estimation-/2024Paper_scripts/env_probs.csv')
cde = pd.read_csv('/home/speersm/GitHub/Linear-Waves-and-Base-Shear-Estimation-/2024Paper_scripts/cond_dens_duo.csv', header=None)

# plot heatmap of cde over env_probs x and y
num_x = 180 
num_y = 90
cde = np.array(cde).reshape(num_x, num_y)
x = np.array(env_probs['x']).reshape(num_x, num_y)
y = np.array(env_probs['y']).reshape(num_x, num_y)

plt.imshow(cde, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto', cmap='hot_r')
plt.colorbar(label='CDE')
plt.xlabel('Hs [m]')
plt.ylabel('Tp [s]')
plt.show()