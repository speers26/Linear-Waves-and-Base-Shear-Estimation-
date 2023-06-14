from wavesim import kinematics as kn
from wavesim import spectrum as spc
import numpy as np

omega_p = 1.5
ss = spc.SeaState(hs=[15], tp=[2*np.pi/omega_p], spctr_type=spc.Jonswap)

z_values = np.linspace(-100, 50, 150)

LinKin = kn.LinearKin(10, period=(120*60), z_values=z_values, sea_state=ss)
LinKin.compute_spectrum()
LinKin.compute_kinematics(cond=False)
eta, _, _, _, _ = LinKin.retrieve_kinematics()

np.savetxt('/home/speersm/GitHub/KHL-data-analysis/data/eta.csv', eta, delimiter=',')
