from wavesim.distest import LoadDistEst
from wavesim.spectrum import SeaState, Jonswap
from wavesim.loading import MorisonLoad
import numpy as np

num_sea_states = 1000
hs = np.tile(15, num_sea_states)
tp = np.tile(10, num_sea_states)
z_values = np.linspace(-100, 50, 150)

ss = SeaState(hs=hs, tp=tp, spctr_type=Jonswap)

np.random.seed(1)

loadEst = LoadDistEst(sea_state=ss, z_values=z_values)
loadEst.compute_cond_crests()
loadEst.compute_kinematics()
loadEst.compute_load()
loadEst.compute_sea_state_max()
loadEst.compute_is_distribution()
loadEst.plot_distribution()
loadEst.plot_distribution(log=False)
