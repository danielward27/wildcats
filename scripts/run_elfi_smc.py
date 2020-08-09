import numpy as np
from sim.model import run_sim, run_sim_vec
from sim.sum_stats import elfi_summary
import elfi
import scipy.stats
import time

start_time = time.time()

elfi.set_client('ipyparallel')

# POD
y_obs = run_sim(
    length=int(5e6),
    recombination_rate=1.8e-8,
    mutation_rate=6e-8,
    pop_size_domestic_1=50,
    pop_size_wild_1=50,
    pop_size_captive=50,
    captive_time=20,
    mig_rate_captive=0.01,
    mig_length_wild=20,
    mig_rate_wild=0.01,
    pop_size_domestic_2=200,
    pop_size_wild_2=200,
    div_time=30000,
    mig_rate_post_split=0.1,
    mig_length_post_split=5000,
    bottleneck_time_wild=3000,
    bottleneck_strength_wild=10000,
    bottleneck_time_domestic=3000,
    bottleneck_strength_domestic=10000,
    random_state=np.random.RandomState(3),
)

# Test priors
length = elfi.Constant(int(5e6))
recombination_rate = elfi.Constant(1.8e-8)
mutation_rate = elfi.Constant(6e-8)
pop_size_domestic_1 = elfi.Prior(scipy.stats.uniform, 50, 500-50)
pop_size_wild_1 = elfi.Prior(scipy.stats.uniform, 50, 500-50)
pop_size_captive = elfi.Prior(scipy.stats.lognorm, 0.3, 0, np.exp(4.5))
mig_rate_captive = elfi.Prior(scipy.stats.beta, 1.2, 40)
captive_time = elfi.Prior(scipy.stats.lognorm, 0.7, 0, np.exp(3))
mig_length_wild = elfi.Prior(scipy.stats.lognorm, 0.7, 0, np.exp(3))
mig_rate_wild = elfi.Prior(scipy.stats.beta(2, 50))
pop_size_domestic_2 = elfi.Prior(scipy.stats.uniform, 1000, 20000-1000)
pop_size_wild_2 = elfi.Prior(scipy.stats.lognorm, 0.2, 0, np.exp(8.8))
bottleneck_strength_domestic = elfi.Prior(scipy.stats.uniform, 0, 40000-0)
bottleneck_time_domestic = elfi.Prior(scipy.stats.norm, 3500, 600)
bottleneck_strength_wild = elfi.Prior(scipy.stats.uniform, 0, 40000-0)
bottleneck_time_wild = elfi.Prior(scipy.stats.norm, 3500, 600)
mig_length_post_split = elfi.Prior(scipy.stats.uniform, 0, 10000-0)
mig_rate_post_split = elfi.Prior(scipy.stats.truncnorm, 0, 1, 0, 0.1)
div_time = elfi.Prior(scipy.stats.norm, 40000, 5000)

# Simulator node
y = elfi.Simulator(run_sim_vec,
                   length, recombination_rate, mutation_rate, pop_size_domestic_1, pop_size_wild_1,
                   pop_size_captive, mig_rate_captive, mig_length_wild, mig_rate_wild,
                   captive_time, pop_size_domestic_2, pop_size_wild_2, div_time, mig_rate_post_split,
                   mig_length_post_split, bottleneck_time_wild, bottleneck_strength_wild,
                   bottleneck_time_domestic, bottleneck_strength_domestic, name="simulator", observed=y_obs)

# Summary node
s = elfi.Summary(elfi_summary, y)

# Distance
d = elfi.Distance('euclidean', s, name="Euclidean distance")
log_d = elfi.Operation(np.log, d)

rej = elfi.Rejection(log_d, batch_size=2, seed=1)
rej_results = rej.sample(n_samples=2, n_sim=8)
print("completed in: {:.2}s".format(time.time()-start_time))
print(rej_results.sample_means_summary())

