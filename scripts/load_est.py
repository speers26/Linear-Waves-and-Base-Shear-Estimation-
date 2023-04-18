from wavesim.loading import LoadDistEst
import matplotlib.pyplot as plt
import numpy as np

hs = 15
tp = 12
num_sea_states = 1000
sea_state_hours = 3
z_values = np.linspace(-100, 50, 150)

np.random.seed(1)

loadEst = LoadDistEst(hs=hs, tp=tp, num_sea_states=num_sea_states, sea_state_hours=sea_state_hours, z_values=z_values)

loadEst.compute_tf_values()
loadEst.compute_spectrum()
loadEst.compute_cond_crests()

loadEst.simulate_sea_states()

X = np.linspace(0, 2*hs, 100)
loadEst.compute_crest_dist(X)
loadEst.compute_load_dist()

loadEst.plot_crest_dist(log=True)
loadEst.plot_load_dist(log=False)
loadEst.plot_load_dist(log=True)

plt.figure()
plt.scatter(loadEst.cond_crests, loadEst.max_load)
plt.xlabel("Crest height")
plt.ylabel("log10(1-P)")
plt.title("Force [MN]")
plt.show()

