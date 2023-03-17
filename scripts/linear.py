from wavesim import spectrum as spctr
from wavesim import kinematics as kin
from wavesim import loading as load
import numpy as np

hs = 30.
tp = 12.
a = 25.
depth = 100
cond = False

z_num = 150
z_range = np.linspace(-depth, 50, z_num)

freq = 1.00  # 3. / (2*np.pi)
period = 100  # total time range
nT = np.floor(period*freq)  # number of time points to evaluate

dt = 1/freq  # time step is determined by frequency
t_range = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt  # centering time around 0

f_range = np.linspace(1e-3, nT - 1, int(nT)) / (nT / freq)  # selecting frequency range from 0 to freq

spectrum1 = spctr.Jonswap(frequency=f_range, hs=hs, tp=tp)
spectrum1.compute_density()

lin_wave = kin.LinearKin(t_values=t_range, z_values=z_range, spctr=spectrum1)
lin_wave.compute_kinematics(cond=cond, a=a)
lin_wave.plot_kinematics()

lin_load = load.MorisonLoad(lin_wave)
lin_load.compute_load()
lin_load.plot_load()
