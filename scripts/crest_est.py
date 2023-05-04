from wavesim.distest import CrestDistEst
from wavesim.spectrum import SeaState, Jonswap
import matplotlib.pyplot as plt
import numpy as np

num_sea_states = 500
hs = np.tile(15, num_sea_states)
tp = np.tile(10, num_sea_states)
z_values = np.linspace(-100, 50, 150)

ss = SeaState(hs=hs, tp=tp, spctr_type=Jonswap)

np.random.seed(1)

crestEst = CrestDistEst(sea_state=ss, z_values=z_values)
crestEst.compute_cond_crests()
crestEst.compute_kinematics()
crestEst.compute_sea_state_max()
crestEst.compute_is_distribution()
crestEst.plot_distribution()
