import numpy as np
import matplotlib.pyplot as plt
import wavesim_functions as wave


if __name__ == "__main__":

    np.random.seed(1)
    write = True
    write_con = True

    hs = 30
    tp = 12
    depth = 100
    cond = False
    a = 0

    z_num = 150
    z_range = np.linspace(-depth, 50, z_num)
    dz = z_range[1] - z_range[0]

    num_sea_states = 2000
    sea_state_hours = 1
    period = 60**2 * sea_state_hours  # total time range in seconds
    waves_per_state = 60**2/tp

    freq = 1.00  # number of sample points per second
    nT = np.floor(period*freq)  # number of time points to evaluate
    t_num = int(nT)  # to work with rest of the code

    dt = 1/freq  # time step is determined by frequency
    t_range = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt  # centering time around 0

    f_range = np.linspace(1e-3, nT - 1, int(nT)) / (nT / freq)  # selecting frequency range from 0 to freq
    om_range = f_range * (2*np.pi)

    # get crest cdf
    CoH = np.linspace(1e-3, 1.5)
    crest_cdf = wave.rayleigh_cdf(CoH * hs, hs)

    # get jonswap densities
    jnswp_dens = wave.djonswap(f_range, hs, tp)

    if write:

        eta = np.empty(t_num)
        u_x = np.empty((t_num, z_num))
        u_z = np.empty((t_num, z_num))
        du_x = np.empty((t_num, z_num))
        du_z = np.empty((t_num, z_num))
        F = np.empty((int(t_num*num_sea_states), z_num))

        for i in range(num_sea_states):
            print(i)
            eta, u_x, u_z, du_x, du_z = wave.fft_random_wave_sim(z_range, depth, a, om_range, jnswp_dens, cond)
            for i_t, t in enumerate(t_range):
                for i_z, z in enumerate(z_range):
                    # eta[i_t], u_x[i_t, i_z], u_z[i_t, i_z], du_x[i_t, i_z], du_z[i_t, i_z] = wave.ptws_random_wave_sim(t=t, z=z, depth=depth, a=a, om_range=om_range, spctrl_dens=jnswp_dens, cond=cond)
                    F[i_t + i * t_num, i_z] = wave.morison_load(u_x[i_t, i_z], du_x[i_t, i_z])

        base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N
        np.savetxt('load.txt', base_shear, delimiter=' ')

    else:

        base_shear = np.loadtxt('load.txt')

    new_t_range = np.linspace(0, period * num_sea_states, int(nT*num_sea_states))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(new_t_range, base_shear, '-k')
    plt.ylabel('Force [MN]')
    plt.xlabel('Time')

    max_forces = np.empty(num_sea_states)
    max_ind = np.empty(num_sea_states)

    for i_s in range(num_sea_states):
        slice = base_shear[i_s*t_num:t_num*(i_s+1)]
        max_forces[i_s] = max(slice)
        slice_ind = np.where(slice == max(slice))[0]
        max_ind[i_s] = int(i_s*t_num + slice_ind)

    colors = np.tile('#FF0000', num_sea_states)

    plt.scatter(max_ind, max_forces, c=colors)
    plt.subplot(2, 1, 2)

    long_emp = np.empty(num_sea_states)
    s_max_forces_0 = np.sort(max_forces)
    for i_f, f in enumerate(s_max_forces_0):
        long_emp[i_f] = sum(max_forces < f )/num_sea_states

    plt.plot(s_max_forces_0, long_emp, '-k')
    plt.show()

    # now do for conditional sim / IS method
    cond = True

    CoHmin = 0
    CoHmax = 2
    CoHnum = num_sea_states
    CoH = np.random.uniform(low=CoHmin, high=CoHmax, size=CoHnum)
    g = 1/(CoHmax-CoHmin)

    period = 60*2  # will simulate sea states of 2 minutes
    waves_per_state = 60*2/tp
    sims_per_state = sea_state_hours * 60 / 2

    f_0 = wave.rayleigh_pdf(CoH*hs, hs)
    fog = f_0 / g

    freq = 1.00  # number of sample points per second
    nT = np.floor(period*freq)  # number of time points to evaluate
    t_num = int(nT)  # to work with rest of the code

    dt = 1/freq  # time step is determined by frequency
    t_range = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt  # centering time around 0

    f_range = np.linspace(1e-3, nT - 1, int(nT)) / (nT / freq)  # selecting frequency range from 0 to freq
    om_range = f_range * (2*np.pi)

    jnswp_dens = wave.djonswap(f_range, hs, tp)

    if write_con:

        eta = np.empty(t_num)
        u_x = np.empty((t_num, z_num))
        u_z = np.empty((t_num, z_num))
        du_x = np.empty((t_num, z_num))
        du_z = np.empty((t_num, z_num))
        F = np.empty((int(t_num*num_sea_states), z_num))

        for i in range(num_sea_states):
            print(i)
            a = CoH[i]
            eta, u_x, u_z, du_x, du_z = wave.fft_random_wave_sim(z_range, depth, a, om_range, jnswp_dens, cond)
            for i_t, t in enumerate(t_range):
                for i_z, z in enumerate(z_range):
                    # eta[i_t], u_x[i_t, i_z], u_z[i_t, i_z], du_x[i_t, i_z], du_z[i_t, i_z] = wave.ptws_random_wave_sim(t=t, z=z, depth=depth, a=a, om_range=om_range, spctrl_dens=jnswp_dens, cond=cond)
                    F[i_t + i * t_num, i_z] = wave.morison_load(u_x[i_t, i_z], du_x[i_t, i_z])

        base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N
        np.savetxt('load_con.txt', base_shear, delimiter=' ')

    else:

        base_shear = np.loadtxt('load_con.txt')

    new_t_range = np.linspace(0, period * num_sea_states, int(nT*num_sea_states))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(new_t_range, base_shear, '-k')
    plt.ylabel('Force [MN]')
    plt.xlabel('Time')

    for i_s in range(num_sea_states):
        slice = base_shear[i_s*t_num:t_num*(i_s+1)]
        max_forces[i_s] = max(slice)
        slice_ind = np.where(slice == max(slice))[0]
        max_ind[i_s] = int(i_s*t_num + slice_ind)

    s_max_forces = np.sort(max_forces)
    is_0 = np.empty(len(s_max_forces))
    for i_f, f in enumerate(s_max_forces_0):
        is_0[i_f] = sum((max_forces < f)*(fog))/sum(fog)

    is_1 = is_0**sims_per_state

    plt.scatter(max_ind, max_forces, c=colors)
    plt.subplot(2, 1, 2)

    plt.plot(s_max_forces_0, is_1, '-b')
    plt.plot(s_max_forces_0, long_emp, '-r')
    plt.show()