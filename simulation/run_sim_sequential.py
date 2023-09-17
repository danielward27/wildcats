from sim import WildcatModel
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
array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

df = pd.read_csv("params_r3.csv")
params = list(df.iloc[array_id])

prior_dict = dict(zip(priors.keys(),params))
thetas = pd.DataFrame(prior_dict, index=['i', ])
filename = "./output/thetas/theta%s.pickle" % array_id

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

summary_stats(data)


with open(filename, 'wb') as handle:
    pickle.dump(thetas, handle, protocol=pickle.DEFAULT_PROTOCOL)

filename1 = "./output/times/time%s.pickle" % array_id

with open(filename1, 'wb') as handle:
    pickle.dump(time, handle, protocol=pickle.DEFAULT_PROTOCOL)


print("### Simulation finished ###")
