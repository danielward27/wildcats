from sim_map import WildcatModel
from priors import priors
import random
import pandas as pd
import numpy as np
from summary import summary_stats
import os
import pickle
import time

print("### Simulation starting ###")
rand = random.randint(1,999999)

params = [1.48969441e+01, 2.90831157e+05, 3.80766122e+03, 4.51339154e+03,
        4.27261249e-02, 5.48782886e-02, 1.81708928e+01, 1.09169148e+02,
        1.24069555e+05, 4.92387189e+03, 9.32893118e+04, 6.82929725e+03,
        7.38126525e+03, 1.51745624e+04, 9.38917107e-09, 1.66682815e-08]

prior_dict = dict(zip(priors.keys(),params))

print(prior_dict)

model = WildcatModel(seq_length=int(45e6), recombination_rate=prior_dict["recombination_rate"], mutation_rate=prior_dict["mutation_rate"])

data, time =  model.simulate(
        captive_time=prior_dict["captive_time"],
        div_time=prior_dict["div_time"],
        div_time_dom=prior_dict["div_time_dom"],
        div_time_scot=prior_dict["div_time_scot"],
        mig_rate_captive=prior_dict["mig_rate_captive"],
        mig_rate_scot=prior_dict["mig_rate_scot"],
        mig_length_scot=prior_dict["mig_length_scot"],
        pop_size_captive=prior_dict["pop_size_captive"],
        pop_size_domestic_1=prior_dict["pop_size_domestic_1"],
        pop_size_lyb_1=prior_dict["pop_size_lyb_1"],
        pop_size_lyb_2=prior_dict["pop_size_lyb_2"],
        pop_size_scot_1=prior_dict["pop_size_scot_1"],
        pop_size_eu_1=prior_dict["pop_size_eu_1"],
        pop_size_eu_2=prior_dict["pop_size_eu_2"],
        n_samples=[6, 65, 22, 15, 4],
        seed=rand)


print("### Simulation finished ###")
