from wavesim.distest import MorisonDistEst
from wavesim.spectrum import SeaState, Jonswap
import numpy as np
import matplotlib.pyplot as plt
import time

num_sea_states = 2000
z_num = 50
hs = np.tile(25, num_sea_states)
tp = np.tile(15, num_sea_states)
z_values = np.linspace(-100, 50, z_num)
c_m = np.tile(1, z_num)
c_d = np.tile(1, z_num)
X = np.linspace(-5, 20, num=1000)

ss = SeaState(hs=hs, tp=tp, spctr_type=Jonswap)

np.random.seed(1)

n_rep = 1

for k in range(n_rep):

    # print(k)
    start = time.time()
    loadEst = MorisonDistEst(sea_state=ss, z_values=z_values, c_d=c_d, c_m=c_m)
    loadEst.compute_cond_crests()
    loadEst.compute_kinematics()
    loadEst.compute_load()
    loadEst.compute_sea_state_max()
    end = time.time()
    print("max series done in " + str(end-start) + " seconds")

    start = time.time()
    loadEst.compute_pdf()
    end = time.time()
    print("pdf done in " + str(end-start) + " seconds")

    start = time.time()
    loadEst.compute_cdf()
    end = time.time()
    print("cdf done in " + str(end-start) + " seconds")

    # int_check = quad(loadEst.eval_pdf, -5, 20)
    # print(int_check)

    start = time.time()
    plt.plot(X, loadEst.eval_pdf(X, smooth=False))
    end = time.time()
    print("plotting non smoothed done in " + str(end-start) + " seconds")

    start = time.time()
    plt.plot(X, loadEst.eval_pdf(X, smooth=True))
    end = time.time()
    print("plotting smoothed done in " + str(end-start) + " seconds")

    plt.xlabel("force [MN]")
    plt.ylabel("pdf")

plt.show()


# plt.plot(X, loadEst.eval_pdf(X))
# plt.show()

# print(loadEst.eval_pdf(np.array([1, 2])))

# print(np.sum(loadEst.dx * loadEst.pdf))
