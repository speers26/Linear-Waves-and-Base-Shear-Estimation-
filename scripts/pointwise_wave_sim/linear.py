from wavesim import spectrum as spctr
from wavesim import kinematics as kin
from wavesim import loading as load
import numpy as np
import matplotlib.pyplot as plt

a = np.array([25])
depth = 100
cond = True
ss1 = spctr.SeaState(hs=np.array([15]), tp=np.array([10]), spctr_type=spctr.Jonswap)

np.random.seed(1)

z_num = 150
z_range = np.linspace(-depth, 50, z_num)
freq = 4.00  # 3. / (2*np.pi)
period = 100  # total time range

lin_wave = kin.LinearKin(sample_f=4.00, period=100, z_values=z_range, sea_state=ss1)
lin_wave.compute_spectrum()
lin_wave.compute_kinematics(cond=cond, a=a)
lin_wave.plot_kinematics()

lin_load = load.MorisonLoad(lin_wave)

plt.subplot(1, 2, 1)
plt.plot(lin_load.kinematics.z_values, lin_load.c_m)
plt.title("c_m")

plt.subplot(1, 2, 2)
plt.plot(lin_load.kinematics.z_values, lin_load.c_d)
plt.title("c_d")
plt.show()

lin_load.compute_load()
lin_load.plot_load(s=[0])
