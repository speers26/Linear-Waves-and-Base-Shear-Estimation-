import numpy as np
import matplotlib.pyplot as plt
import wavesim_functions as wave

if __name__ == "__main__":

    # we will propagate a random wave and its kinematics at a fixed point x=0

    # set sea state and conditioned peak here
    hs = 10.
    tp = 12.
    a = 25.
    depth = 100
    cond = True

    z_num = 150
    z_range = np.linspace(-depth, 50, z_num)
    dz = z_range[1] - z_range[0]

    freq = 1.00  # 3. / (2*np.pi)
    period = 100  # total time range
    nT = np.floor(period*freq)  # number of time points to evaluate
    t_num = int(nT)  # to work with rest of the code

    dt = 1/freq  # time step is determined by frequency
    t_range = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt  # centering time around 0

    f_range = np.linspace(1e-3, nT - 1, int(nT)) / (nT / freq)  # selecting frequency range from 0 to freq
    om_range = f_range * (2*np.pi)
    # all above taken from rand_wave_sim.py

    jnswp_dens = wave.djonswap(f_range, hs, tp)

    eta_fft, u_x_fft, u_z_fft, du_x_fft, du_z_fft = wave.fft_random_wave_sim(z_range, depth, a, om_range, jnswp_dens, cond)

    F = np.empty((t_num, z_num))
    for i_t, t in enumerate(t_range):
        for i_z, z in enumerate(z_range):
            F[i_t, i_z] = wave.morison_load(u_x_fft[i_t, i_z], du_x_fft[i_t, i_z])
    base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N

    z_grid, t_grid = np.meshgrid(z_range, t_range)

    plt.figure()

    plt.subplot(2, 2, 1)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=u_x_fft.flatten())
    plt.plot(t_range, eta_fft[0], '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('u')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=u_z_fft.flatten())
    plt.plot(t_range, eta_fft[0], '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('v')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=du_x_fft.flatten())
    plt.plot(t_range, eta_fft[0], '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('du')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=du_z_fft.flatten())
    plt.plot(t_range, eta_fft[0], '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('dv')
    plt.colorbar()

    plt.figure()
    plt.plot(t_grid, base_shear)
    plt.ylabel('Force [MN]')
    plt.xlabel('Time')

    plt.show()

    # run below to show wheeler stretching for non-fft

    eta = np.empty(t_num)
    u_x = np.empty((t_num, z_num))
    u_z = np.empty((t_num, z_num))
    du_x = np.empty((t_num, z_num))
    du_z = np.empty((t_num, z_num))
    F = np.empty((t_num, z_num))

    for i_t, t in enumerate(t_range):
        for i_z, z in enumerate(z_range):
            eta[i_t], u_x[i_t, i_z], u_z[i_t, i_z], du_x[i_t, i_z], du_z[i_t, i_z] = wave.ptws_random_wave_sim(t=t, z=z, depth=depth, a=a, om_range=om_range, spctrl_dens=jnswp_dens, cond=cond)
            F[i_t, i_z] = wave.morison_load(u_x[i_t, i_z], du_x[i_t, i_z])
    # F = xr.DataArray(F, dims=["t", "z"])

    base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=u_x.flatten())
    plt.ylim([-depth, 50])
    plt.plot(t_range, eta, '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('u')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=u_z.flatten())
    plt.plot(t_range, eta, '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('v')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=du_x.flatten())
    plt.plot(t_range, eta, '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('du')
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.scatter(t_grid.flatten(), z_grid.flatten(), s=1, c=du_z.flatten())
    plt.plot(t_range, eta, '-k')
    plt.xlabel('time')
    plt.ylabel('depth')
    plt.title('dv')
    plt.colorbar()

    plt.figure()
    plt.plot(t_grid, base_shear)
    plt.ylabel('Force [MN]')
    plt.xlabel('Time')

    plt.figure()
    plt.plot(t_grid, eta, '-k')
    plt.plot(t_grid, eta_fft[0], '--r')

    plt.show()
