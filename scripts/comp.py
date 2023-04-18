
import wavesim.kinematics as kin
import wavesim.loading as load
import wavesim.spectrum as spctr
import numpy as np
import matplotlib.pyplot as plt

hs = 15
tp = 12
num_sea_states = 100
sea_state_hours = 3
z_values = np.linspace(-100, 50, 150)

freq = 1.00  # 3. / (2*np.pi)
period = 100  # total time range
nT = np.floor(period*freq)  # number of time points to evaluate

dt = 1/freq  # time step is determined by frequency
t_range = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt  # centering time around 0
f_range = np.linspace(1e-3, nT - 1, int(nT)) / (nT / freq)  # selecting frequency range from 0 to freq

spectrum1 = spctr.Jonswap(frequency=f_range, hs=hs, tp=tp)
spectrum1.compute_density()

r_crests = np.sort(np.random.uniform(low=0, high=2*hs, size=100))

max_load = np.empty(100)

for i_c, c in enumerate(r_crests):
    lin_wave = kin.LinearKin(t_values=t_range, z_values=z_values, spctr=spectrum1)
    lin_wave.compute_kinematics(cond=True, a=c)

    loading = load.MorisonLoad(lin_wave)
    loading.compute_load()
    max_load[i_c] = np.max(loading.load)

plt.figure()
plt.plot(r_crests, max_load)
plt.show()
