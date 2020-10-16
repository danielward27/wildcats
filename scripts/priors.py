# Script defines and plots the prior distributions. Priors are pickled.
# Prior distributions. First defined distribution is sampled by ELFI,
# The second distribution is the target/scaled up distribution used by the simulator
# This avoids ill-conditioned/singular matrix error.

import numpy as np
import scipy.stats
from sim.utils import ScaledDist
import matplotlib.pyplot as plt
import pickle

priors = {
    "bottleneck_strength_domestic": ScaledDist(scipy.stats.truncnorm(a=-0.5, b=np.inf, loc=0, scale=1),
                                               scipy.stats.truncnorm(a=-0.5, b=np.inf, loc=7500, scale=15000)),
    "bottleneck_strength_wild": ScaledDist(scipy.stats.truncnorm(a=-0.5, b=np.inf, loc=0, scale=1),
                                           scipy.stats.truncnorm(a=-0.5, b=np.inf, loc=7500, scale=15000)),
    "bottleneck_time_domestic": ScaledDist(scipy.stats.truncnorm(a=-6, b=np.inf, loc=0, scale=1),
                                           scipy.stats.truncnorm(a=-6, b=np.inf, loc=3500, scale=500)),
    "bottleneck_time_wild": ScaledDist(scipy.stats.truncnorm(a=-6, b=np.inf, loc=0, scale=1),
                                       scipy.stats.truncnorm(a=-6, b=np.inf, loc=3500, scale=500)),
    "captive_time": ScaledDist(scipy.stats.lognorm(s=0.4, loc=0, scale=1),
                               scipy.stats.lognorm(s=0.4, loc=1, scale=np.exp(2.7))),
    "div_time": ScaledDist(scipy.stats.norm(loc=0, scale=1),
                           scipy.stats.norm(loc=40000, scale=4000)),
    "mig_length_post_split": ScaledDist(scipy.stats.uniform(loc=0, scale=1),
                                        scipy.stats.uniform(loc=0, scale=10000)),
    "mig_length_wild": ScaledDist(scipy.stats.lognorm(s=0.4, loc=0, scale=1),
                                  scipy.stats.lognorm(s=0.4, loc=1, scale=np.exp(2.5))),
    "mig_rate_captive": ScaledDist(scipy.stats.lognorm(s=0.5, loc=0, scale=1),
                                   scipy.stats.lognorm(s=0.5, loc=0, scale=0.08)),
    "mig_rate_post_split": ScaledDist(scipy.stats.truncnorm(a=0, b=5, loc=0, scale=1),
                                      scipy.stats.truncnorm(a=0, b=5, loc=0, scale=0.2)),
    "mig_rate_wild": ScaledDist(scipy.stats.lognorm(s=0.5, loc=0, scale=1),
                                scipy.stats.lognorm(s=0.5, loc=0, scale=0.08)),
    "pop_size_wild_1": ScaledDist(scipy.stats.lognorm(s=0.2, loc=0, scale=1),
                                  scipy.stats.lognorm(s=0.2, loc=30, scale=np.exp(8.7))),
    "pop_size_wild_2": ScaledDist(scipy.stats.lognorm(s=0.2, loc=0, scale=1),
                                  scipy.stats.lognorm(s=0.2, loc=30, scale=np.exp(9))),
    "pop_size_captive": ScaledDist(scipy.stats.lognorm(s=0.5, loc=0, scale=1),
                                   scipy.stats.lognorm(s=0.5, loc=10, scale=100)),
    "pop_size_domestic_1": ScaledDist(scipy.stats.lognorm(s=0.25, loc=0, scale=1),
                                      scipy.stats.lognorm(s=0.25, loc=5, scale=np.exp(8.75))),
    "pop_size_domestic_2": ScaledDist(scipy.stats.lognorm(s=0.2, loc=0, scale=1),
                                      scipy.stats.lognorm(s=0.2, loc=30, scale=np.exp(9.2)))
}

for prior_name, prior in priors.items():
    prior.plot(x_lab=prior_name)
    plt.savefig("../plots/priors/{}.png".format(prior_name))
    plt.clf()

with open("../output/priors.pkl", "wb") as f:
    pickle.dump(priors, f)
