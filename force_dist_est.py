import numpy as np
import matplotlib.pyplot as plt
import wavesim_functions as wave


if __name__ == "__main__":

    np.random.seed(12345)
    write = True
    write_con = True

    # set up wave conditions
    hs = 25
    tp = 12
    depth = 100
    cond = False
    a = 0

    # set up arrays
    z_num = 150
    z_range = np.linspace(-depth, 50, z_num)
    dz = z_range[1] - z_range[0]

    num_sea_states = 2000
    sea_state_hours = 1
    period = 60**2 * sea_state_hours  # total time range in seconds
    waves_per_state = period/tp

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

    # get jonswap density
    jnswp_dens = wave.djonswap(f_range, hs, tp)

    # if we want new wave data
    if write:
        # set up arrays (don't actually need to do this for all)
        eta = np.empty(int(t_num*num_sea_states))
        u_x = np.empty((int(t_num*num_sea_states), z_num))
        u_z = np.empty((t_num, z_num))
        du_x = np.empty((t_num, z_num))
        du_z = np.empty((t_num, z_num))
        F = np.empty((int(t_num*num_sea_states), z_num))

        # populate arrays
        for i in range(num_sea_states):
            print(i)
            eta[i*t_num:(i+1)*t_num], u_x, u_z, du_x, du_z = wave.fft_random_wave_sim(z_range, depth, a, om_range, jnswp_dens, cond)
            for i_t, t in enumerate(t_range):
                for i_z, z in enumerate(z_range):
                    F[i_t + i * t_num, i_z] = wave.morison_load(u_x[i_t, i_z], du_x[i_t, i_z])
        base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N

        # write data to text files
        np.savetxt('load.txt', base_shear, delimiter=' ')
        np.savetxt('eta.txt', eta, delimiter=' ')
        # np.savetxt('h_velocity.txt', u_x, delimiter=' ')

    # if we want to use the old wave data
    else:
        # read data from the text files
        eta = np.loadtxt('eta.txt')
        base_shear = np.loadtxt('load.txt')
        # u_x = np.loadtxt('h_velocity.txt')

    # extend the time range to include all sea states
    new_t_range = np.linspace(0, period * num_sea_states, int(nT*num_sea_states))

    # get maximum force in each sea state
    max_forces = np.empty(num_sea_states)
    max_ind = np.empty(num_sea_states)
    for i_s in range(num_sea_states):
        slice = base_shear[i_s*t_num:t_num*(i_s+1)]
        max_forces[i_s] = max(slice)
        slice_ind = np.where(slice == max(slice))[0]
        max_ind[i_s] = int(i_s*t_num + slice_ind)

    # get empirical distribution of max forces
    long_emp = np.empty(num_sea_states)
    s_max_forces_0 = np.sort(max_forces)
    for i_f, f in enumerate(s_max_forces_0):
        long_emp[i_f] = sum(max_forces < f)/num_sea_states

    # set colors for scatter plots
    colors = np.tile('#FF0000', num_sea_states)

    # plot max forces ts and emp distribution
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(new_t_range, base_shear, '-k')
    plt.ylabel('Force [MN]')
    plt.xlabel('Time')
    plt.scatter(max_ind, max_forces, c=colors)
    plt.subplot(2, 1, 2)
    plt.plot(s_max_forces_0, long_emp, '-k')

    # get max crest from each sea state
    max_crests = np.empty(num_sea_states)
    for i_s in range(num_sea_states):
        slice = eta[i_s*t_num:t_num*(i_s+1)]
        max_crests[i_s] = max(slice)
        slice_ind = np.where(slice == max(slice))[0]
        max_ind[i_s] = int(i_s*t_num + slice_ind)

    # get max crest height empirical distribution
    long_crest_emp = np.empty(num_sea_states)
    s_max_crests_0 = np.sort(max_crests)
    for i_c, c in enumerate(s_max_crests_0):
        long_crest_emp[i_c] = sum(max_crests < c)/num_sea_states

    # get true crest height distribution (rayleigh)
    true_crest_dist = wave.rayleigh_cdf(s_max_crests_0, hs)**waves_per_state

    # plot crest height ts and emp distribution with rayleigh
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(new_t_range, eta, '-k')
    plt.ylabel('Crest Height [m]')
    plt.xlabel('Time')
    plt.scatter(max_ind, max_crests, c=colors)
    plt.subplot(2, 1, 2)
    plt.plot(s_max_crests_0, long_crest_emp, '-k')
    plt.plot(s_max_crests_0, true_crest_dist, '--g')

    # now do for conditional sim / IS method
    cond = True

    # get proposal sample and density of crest
    CoHmin = 0
    CoHmax = 2
    CoHnum = num_sea_states
    CoH = np.random.uniform(low=CoHmin, high=CoHmax, size=CoHnum)
    g = 1/(CoHmax-CoHmin)

    # will simulate sea states of 2 minutes
    sea_state_minutes = 2
    period = 60*sea_state_minutes
    waves_per_state = period/tp
    sims_per_state = sea_state_hours * 60 / 2

    # get true density and weights
    c = CoH * hs
    f_0 = wave.rayleigh_pdf(CoH*hs, hs)
    fog = f_0/g

    # ----- do this if true density needs changing from regular rayleigh (not sure if it does or not) ----- #
    # f_prime = np.exp(16/hs**2 - (16*c/hs**2)**2)
    # f = waves_per_state * f_0**(waves_per_state-1) * f_prime
    # fog = f / g

    # redo arrays
    freq = 1.00  # number of sample points per second
    nT = np.floor(period*freq)  # number of time points to evaluate
    t_num = int(nT)  # to work with rest of the code

    dt = 1/freq  # time step is determined by frequency
    t_range = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt  # centering time around 0

    f_range = np.linspace(1e-3, nT - 1, int(nT)) / (nT / freq)  # selecting frequency range from 0 to freq
    om_range = f_range * (2*np.pi)

    # redo jonswap density
    jnswp_dens = wave.djonswap(f_range, hs, tp)

    if write_con:
        # generate wave data and write to text files
        eta = np.empty(int(t_num*num_sea_states))
        u_x = np.empty((t_num, z_num))
        u_z = np.empty((t_num, z_num))
        du_x = np.empty((t_num, z_num))
        du_z = np.empty((t_num, z_num))
        F = np.empty((int(t_num*num_sea_states), z_num))

        for i in range(num_sea_states):
            print(i)
            a = CoH[i]
            eta[i*t_num:(i+1)*t_num], u_x, u_z, du_x, du_z = wave.fft_random_wave_sim(z_range, depth, a, om_range, jnswp_dens, cond)
            for i_t, t in enumerate(t_range):
                for i_z, z in enumerate(z_range):
                    F[i_t + i * t_num, i_z] = wave.morison_load(u_x[i_t, i_z], du_x[i_t, i_z])

        base_shear = np.sum(F, axis=1) * dz / 1e6  # 1e6 converts to MN from N
        np.savetxt('load_con.txt', base_shear, delimiter=' ')
        np.savetxt('eta_con.txt', eta, delimiter=' ')
        # np.savetxt('h_v_con.txt', u_x, delimiter=' ')

    else:
        # read wave date from txt files
        base_shear = np.loadtxt('load_con.txt')
        eta = np.loadtxt('eta_con.txt')
        # u_x = np.loadtxt('h_v_con.txt')

    new_t_range = np.linspace(0, period * num_sea_states, int(nT*num_sea_states))

    # get max forces
    for i_s in range(num_sea_states):
        slice = base_shear[i_s*t_num:t_num*(i_s+1)]
        max_forces[i_s] = max(slice)
        slice_ind = np.where(slice == max(slice))[0]
        max_ind[i_s] = int(i_s*t_num + slice_ind)

    # get IS max force distribution
    s_max_forces = np.sort(max_forces)
    is_0 = np.empty(len(s_max_forces))
    for i_f, f in enumerate(s_max_forces):
        is_0[i_f] = sum((max_forces < f)*(fog))/sum(fog)
    is_1 = is_0**sims_per_state

    # plot max forces and comparison of empirical and IS force dists
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(new_t_range, base_shear, '-k')
    plt.ylabel('Force [MN]')
    plt.xlabel('Time')
    plt.scatter(max_ind, max_forces, c=colors)
    plt.subplot(2, 1, 2)
    plt.plot(s_max_forces, np.log10(1-is_1), '-b')
    plt.plot(s_max_forces_0, np.log10(1-long_emp), '-r')

    # get max crests
    max_crests = np.empty(num_sea_states)
    for i_s in range(num_sea_states):
        slice = eta[i_s*t_num:t_num*(i_s+1)]
        max_crests[i_s] = max(slice)
        slice_ind = np.where(slice == max(slice))[0]
        max_ind[i_s] = int(i_s*t_num + slice_ind)

    # get IS crest distribution
    is_crest_dist = np.empty(num_sea_states)
    s_max_crests = np.sort(max_crests)
    for i_c, c in enumerate(s_max_crests):
        is_crest_dist[i_c] = sum((max_crests < c)*(fog))/sum(fog)
    is_crest_dist_1 = is_crest_dist**sims_per_state

    # plot max crests and emp crest, true crest and IS crest distributions
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(new_t_range, eta, '-k')
    plt.ylabel('Crest Height [m]')
    plt.xlabel('Time')
    plt.scatter(max_ind, max_crests, c=colors)
    plt.subplot(2, 1, 2)
    plt.plot(s_max_crests_0, np.log10(1-long_crest_emp), '-r')
    plt.plot(s_max_crests_0, np.log10(1-true_crest_dist), '--g')
    plt.plot(s_max_crests, np.log10(1-is_crest_dist_1), '-b')
    plt.show()
