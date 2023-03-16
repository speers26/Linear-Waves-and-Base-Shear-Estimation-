import numpy as np
import matplotlib.pyplot as plt
from wavesim import kinematics as kin

if __name__ == '__main__':

    h = 100
    T = 20
    A = 20/2

    n_depth = 151
    z_range = np.linspace(-h, 50, n_depth)

    n_time = 200
    time = np.linspace(-20, 20, n_time)

    airy_wave = kin.AiryKin(t_values=time, z_values=z_range, H=A*2, T=T)
    airy_wave.compute_kinematics()
    airy_wave.plot_kinematics()


    # for i_t, t in enumerate(time):
    #     for i_z, z in enumerate(z_range):
    #         eta[i_t], u[i_t, i_z], w[i_t, i_z], du[i_t, i_z], dw[i_t, i_z] = airy_kinematics(k, h, A, x, omega, t, z)

    #         F[i_t, i_z] = morison_load(u[i_t, i_z], du[i_t, i_z])

    # base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N

    # plt.figure()
    # plt.subplot(2, 2, 1)
    # z_range_grid, time_grid = np.meshgrid(z_range, time)

    # plt.scatter(time_grid.flatten(), z_range_grid.flatten(), s=1, c=u.flatten())
    # plt.plot(time, eta, '-k')
    # plt.title('u')
    # plt.xlabel('time')
    # plt.ylabel('depth')
    # plt.colorbar()

    # plt.subplot(2, 2, 2)
    # plt.scatter(time_grid.flatten(), z_range_grid.flatten(), s=1, c=w.flatten())
    # plt.plot(time, eta, '-k')
    # plt.title('w')
    # plt.xlabel('time')
    # plt.ylabel('depth')
    # plt.colorbar()

    # plt.subplot(2, 2, 3)
    # plt.scatter(time_grid.flatten(), z_range_grid.flatten(), s=1, c=du.flatten())
    # plt.plot(time, eta, '-k')
    # plt.title('du')
    # plt.xlabel('time')
    # plt.ylabel('depth')
    # plt.colorbar()

    # plt.subplot(2, 2, 4)
    # plt.scatter(time_grid.flatten(), z_range_grid.flatten(), s=1, c=dw.flatten())
    # plt.plot(time, eta, '-k')
    # plt.title('dw')
    # plt.xlabel('time')
    # plt.ylabel('depth')
    # plt.colorbar()

    # plt.show()
