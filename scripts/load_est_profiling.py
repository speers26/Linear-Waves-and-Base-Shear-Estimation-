from wavesim.loading import LoadDistEst
import numpy as np
import cProfile

def estimate_load():
    hs = 25
    tp = 10
    num_sea_states = 2000
    sea_state_hours = 2
    z_values = np.linspace(-100, 50, 50)

    np.random.seed(1)

    loadEst = LoadDistEst(hs=hs, tp=tp, num_sea_states=num_sea_states, sea_state_hours=sea_state_hours, z_values=z_values)

    loadEst.compute_tf_values()
    loadEst.compute_spectrum()
    loadEst.compute_cond_crests()

    loadEst.simulate_sea_states()

    X = np.linspace(0, 2*hs, 100)
    loadEst.compute_crest_dist(X)
    loadEst.compute_load_dist()

    loadEst.plot_crest_dist(log=True)
    loadEst.plot_load_dist()


cProfile.run("estimate_load()", "LoadEstProf")
