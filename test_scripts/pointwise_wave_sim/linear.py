from wavesim import spectrum as spctr
from wavesim import kinematics as kin
from wavesim import loading as load
import numpy as np
import matplotlib.pyplot as plt

a = np.array([25])
depth = 100
cond = False
ss1 = spctr.SeaState(hs=np.array([15]), tp=np.array([10]), spctr_type=spctr.Jonswap)

np.random.seed(1)

z_num = 150
z_range = np.linspace(-depth, 50, z_num)
freq = 4.00  # 3. / (2*np.pi)
period = 300  # total time range

lin_wave = kin.LinearKin(sample_f=4.00, period=100, z_values=z_range, sea_state=ss1)
lin_wave.compute_spectrum()
lin_wave.compute_kinematics(cond=cond, a=a)
lin_wave.plot_kinematics()

# pick cd, cdms
cm_l = 1.0
cm_u = 100.0
cd_l = 1.0
cd_u = 100.0
deck_height = 25.0

diffs = abs(lin_wave.z_values-deck_height)
deck_ind = np.where(diffs == np.min(diffs))[0][0]
c_m = np.concatenate((np.tile(cm_l, deck_ind), np.tile(cm_u, 3), np.tile(cm_l, lin_wave.nz-deck_ind-3)))
c_d = np.concatenate((np.tile(cd_l, deck_ind), np.tile(cd_u, lin_wave.nz-deck_ind)))

lin_load = load.MorisonLoad(lin_wave, c_d, c_m)

plt.subplot(1, 2, 1)
plt.plot(lin_load.kinematics.z_values, lin_load.c_m)
plt.title("c_m")

plt.subplot(1, 2, 2)
plt.plot(lin_load.kinematics.z_values, lin_load.c_d)
plt.title("c_d")
plt.show()

lin_load.compute_load()
lin_load.plot_load(s=[0])
