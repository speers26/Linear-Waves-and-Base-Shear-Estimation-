import numpy as np
import matplotlib.pyplot as plt
import wavesim_functions as wave


if __name__ == "__main__":

    np.random.seed(12345)
    write = False
    write_con = False

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

    freq = 4.00  # number of sample points per second
    nT = np.floor(period*freq)  # number of time points to evaluate
    t_num = int(nT)  # to work with rest of the code

    dt = 1/freq  # time step is determined by frequency
    t_range = np.linspace(-nT/2, nT/2 - 1, int(nT)) * dt  # centering time around 0

    f_range = np.linspace(1e-3, nT - 1, int(nT)) / (nT / freq)  # selecting frequency range from 0 to freq
    om_range = f_range * (2*np.pi)

    # get jonswap density
    jnswp_dens = wave.djonswap(f_range, hs, tp)

    # if we want new wave data
    if write:
        # set up arrays (don't actually need to do this for all)
        eta = np.empty([num_sea_states, t_num])

        # populate arrays
        for i in range(num_sea_states):
            print(i)
            eta[i, :], _, _, _, _ = wave.fft_random_wave_sim(z_range, depth, a, om_range, jnswp_dens, cond)

        np.savetxt('eta.txt', eta, delimiter=' ')

    # if we want to use the old wave data
    else:
        # read data from the text files
        eta = np.loadtxt('eta.txt')

    # extend the time range to include all sea states
    new_t_range = np.linspace(0, period * num_sea_states, int(nT*num_sea_states))

    # get max crest from each sea state
    max_crests_0 = np.empty(num_sea_states)
    max_ind = np.empty(num_sea_states)
    for i_s in range(num_sea_states):
        slice = eta[i_s*t_num:t_num*(i_s+1)]
        max_crests_0[i_s] = max(slice)
        slice_ind = np.where(slice == max(slice))[0]
        max_ind[i_s] = int(i_s*t_num + slice_ind)

    # now do for conditional sim
    cond = True

    # get proposal sample and density of crest
    CoHmin = 0
    CoHmax = 2
    CoHnum = num_sea_states
    CoH = np.random.uniform(low=CoHmin, high=CoHmax, size=CoHnum)
    g = 1/((CoHmax-CoHmin)*hs)  # density for crest heights not CoH
    r_crests = np.sort(CoH * hs)

    # get true crest height distribution (rayleigh)
    true_crest_dist = wave.rayleigh_cdf(r_crests, hs)**waves_per_state  # density for crest heights no CoH

    # will simulate sea states of 2 minutes
    sea_state_minutes = 2
    period = 60*sea_state_minutes
    waves_per_state = period/tp
    sims_per_state = sea_state_hours * 60 / sea_state_minutes

    # get weights
    f = wave.rayleigh_pdf(r_crests, hs)
    fog = f/g

    x = np.linspace(0, 2*hs, num=100)
    cdf_crest = np.empty(x.shape)
    for i_x, c in enumerate(x):
        cdf_crest[i_x] = np.sum((r_crests<c) * fog)/np.sum(fog)

    cdf_crest_ss_max = cdf_crest**(waves_per_state*sims_per_state)
    rayleigh_cdf = wave.rayleigh_cdf(x, hs)
    rayleigh_ss_max_cdf = rayleigh_cdf**(waves_per_state*sims_per_state)

    plt.figure()
    plt.plot(r_crests, f)

    # redo arrays
    freq = 4.00  # number of sample points per second
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
        for i in range(num_sea_states):
            print(i)
            a = CoH[i] * hs
            eta[i*t_num:(i+1)*t_num], _, _, _, _ = wave.fft_random_wave_sim(z_range, depth, a, om_range, jnswp_dens, cond)

        np.savetxt('eta_con.txt', eta, delimiter=' ')
        # np.savetxt('h_v_con.txt', u_x, delimiter=' ')

    else:
        # read wave date from txt files
        eta = np.loadtxt('eta_con.txt')

    new_t_range = np.linspace(0, period * num_sea_states, int(nT*num_sea_states))

    # get emp crest dist
    emp_crest = np.empty(num_sea_states)
    for i_c, c in enumerate(r_crests):
        emp_crest[i_c] = sum(max_crests_0 < c)/num_sea_states

    # get cond max crests
    max_crests = np.empty(num_sea_states)
    for i_s in range(num_sea_states):
        slice = eta[i_s*t_num:t_num*(i_s+1)]
        max_crests[i_s] = max(slice)
        slice_ind = np.where(slice == max(slice))[0]
        max_ind[i_s] = int(i_s*t_num + slice_ind)

    # get IS crest distribution
    is_crest_dist = np.empty(x.shape)
    for i_c, c in enumerate(x):
        is_crest_dist[i_c] = np.sum((max_crests < c) * fog)/np.sum(fog)
    is_crest_dist_1 = is_crest_dist**sims_per_state

    colors = np.tile('#FF0000', num_sea_states)

    cdf_crest_two_min = cdf_crest**waves_per_state
    cdf_rayleigh_ss_two_min = rayleigh_cdf**waves_per_state

    plt.figure()
    plt.plot(r_crests, np.sort(max_crests), '.')
    plt.plot([0,50], [0,50], 'k')
    plt.show()

    plt.figure()
    plt.plot(t_range, eta[np.argmax(max_crests)*t_num:(np.argmax(max_crests)+1)*t_num])
    plt.show()

    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(x/hs, np.log10(1-cdf_crest))
    plt.plot(x/hs, np.log10(1-rayleigh_cdf), '-r')

    plt.subplot(1,3,2)
    plt.plot(x/hs, np.log10(1-cdf_crest_two_min))
    plt.plot(x/hs, np.log10(1-cdf_rayleigh_ss_two_min), '-r')
    plt.plot(x/hs, np.log10(1-is_crest_dist), '--g')

    plt.subplot(1,3,3)
    plt.plot(x/hs, np.log10(1-cdf_crest_ss_max))
    plt.plot(x/hs, np.log10(1-rayleigh_ss_max_cdf), '-r')
    plt.plot(x/hs, np.log10(1-is_crest_dist_1), '--g')
    plt.show()


    # plot max crests and emp crest, true crest and IS crest distributions
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(new_t_range, eta, '-k')
    plt.ylabel('Crest Height [m]')
    plt.xlabel('Time')
    plt.scatter(max_ind, max_crests, c=colors)
    plt.subplot(2, 1, 2)
    plt.plot(r_crests/hs, np.log10(1-true_crest_dist), '--g')
    plt.plot(r_crests/hs, np.log10(1-emp_crest), '-r')
    plt.plot(r_crests/hs, np.log10(1-is_crest_dist_1), '-b')
    plt.show()
