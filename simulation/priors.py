# Script defines the prior distributions.

import numpy as np
import scipy.stats
#from utils import ScaledDist
import matplotlib.pyplot as plt
import pickle
import pandas as pd
#from sim.utils import check_params

priors = {
    "captive_time": scipy.stats.lognorm(s=0.4, loc=1, scale=np.exp(2.7)),
    "div_time": scipy.stats.lognorm(s=1, loc=20000, scale=50000),
    "div_time_dom": scipy.stats.lognorm(s=0.2, loc=1000, scale=2500),
    "div_time_scot": scipy.stats.lognorm(s=0.05, loc=1000, scale=3500),
    "mig_rate_captive": scipy.stats.lognorm(s=0.5, loc=0, scale=0.08),
    "mig_rate_scot": scipy.stats.lognorm(s=0.5, loc=0, scale=0.08),
    "mig_length_scot": scipy.stats.lognorm(s=0.4, loc=1, scale=np.exp(2.5)),
    "pop_size_captive": scipy.stats.lognorm(s=0.5, loc=10, scale=100),
    "pop_size_domestic_1": scipy.stats.lognorm(s=1, loc=0, scale=50000),
    "pop_size_lyb_1": scipy.stats.lognorm(s=1, loc=0, scale=20000),
    "pop_size_lyb_2": scipy.stats.lognorm(s=1, loc=0, scale=80000),
    "pop_size_scot_1": scipy.stats.lognorm(s=0.2, loc=30, scale=np.exp(8.7)),
    "pop_size_eu_1": scipy.stats.lognorm(s=1, loc=0, scale=10000),
    "pop_size_eu_2": scipy.stats.lognorm(s=1, loc=0, scale=40000),
    "mutation_rate": scipy.stats.lognorm(s=1, loc=0, scale=1.e-8),
    "recombination_rate": scipy.stats.lognorm(s=0.5, loc=1e-8, scale=5.0e-9),
}