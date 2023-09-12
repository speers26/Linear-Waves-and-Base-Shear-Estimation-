from wavesim.distest import MorisonDistEst
from wavesim.spectrum import SeaState, Jonswap
import numpy as np
import matplotlib.pyplot as plt

num_sea_states = 2000
z_num = 50
hs = np.tile(25, num_sea_states)
tp = np.tile(15, num_sea_states)
z_values = np.linspace(-100, 50, z_num)
c_m = np.tile(1, z_num)
c_d = np.tile(1, z_num)
X = np.linspace(0, 10, num=1000)

ss = SeaState(hs=hs, tp=tp, spctr_type=Jonswap)

np.random.seed(1)

n_rep = 5

for k in range(n_rep):

    print(k)
    loadEst = MorisonDistEst(sea_state=ss, z_values=z_values, c_d=c_d, c_m=c_m)
    loadEst.compute_cond_crests()
    loadEst.compute_kinematics()
    loadEst.compute_load()
    loadEst.compute_sea_state_max()
    loadEst.compute_is_distribution()
    loadEst.compute_density()

    plt.plot(X, loadEst.eval_pdf(X))
    plt.xlabel("force [MN]")
    plt.ylabel("pdf")

plt.show()

# plt.plot(X, loadEst.eval_pdf(X))
# plt.show()

# print(loadEst.eval_pdf(np.array([1, 2])))

# print(np.sum(loadEst.dx * loadEst.pdf))
