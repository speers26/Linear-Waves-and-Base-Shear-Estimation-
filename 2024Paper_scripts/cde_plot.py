import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd

# load csvs
env_probs = pd.read_csv('/home/speersm/GitHub/Linear-Waves-and-Base-Shear-Estimation-/2024Paper_scripts/env_probs.csv')
cde = pd.read_csv('/home/speersm/GitHub/Linear-Waves-and-Base-Shear-Estimation-/2024Paper_scripts/cond_dens_duo.csv')

# plot heatmap of cde
plt.imshow(cde, aspect='auto')
plt.colorbar()
plt.show()
