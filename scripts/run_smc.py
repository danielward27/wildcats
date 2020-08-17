import logging
import numpy as np
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import elfi
from sim.utils import ScaledDist, plot_dist
from sim.model import elfi_sim
from sim.sum_stats import elfi_summary
import scipy.stats
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
elfi.set_client("ipyparallel", profile="pbs")  # Assumes ipython profile named pbs

start_time = time.time()

min_pop = 30  # Truncate population sizes to ensure enough to sample

### Priors ###
priors = {
    "bottleneck_strength_domestic": ScaledDist(scipy.stats.uniform(loc=0, scale=1),
                                               scipy.stats.uniform(loc=0, scale=40000)),
    "bottleneck_strength_wild": ScaledDist(scipy.stats.uniform(loc=0, scale=1),
                                           scipy.stats.uniform(loc=0, scale=40000)),
    "bottleneck_time_domestic": ScaledDist(scipy.stats.truncnorm(a=-6, b=np.inf, loc=0, scale=1),
                                           scipy.stats.truncnorm(a=-6, b=np.inf, loc=3500, scale=500)),  # Truncated at 500 (n slim generations)
    "bottleneck_time_wild": ScaledDist(scipy.stats.truncnorm(a=-6, b=np.inf, loc=0, scale=1),
                                       scipy.stats.truncnorm(a=-6, b=np.inf, loc=3500, scale=500)),  # Truncated at 500 (n slim generations)
    "captive_time": ScaledDist(scipy.stats.lognorm(s=0.7, loc=0, scale=1),
                               scipy.stats.lognorm(s=0.7, loc=1, scale=np.exp(3))),
    "div_time": ScaledDist(scipy.stats.norm(loc=0, scale=1),
                           scipy.stats.norm(loc=40000, scale=4000)),
    "mig_length_post_split": ScaledDist(scipy.stats.uniform(loc=0, scale=1),
                                        scipy.stats.uniform(loc=0, scale=10000)),
    "mig_length_wild": ScaledDist(scipy.stats.lognorm(s=0.7, loc=0, scale=1),
                                  scipy.stats.lognorm(s=0.7, loc=1, scale=np.exp(3))),
    "mig_rate_captive": ScaledDist(scipy.stats.lognorm(s=0.5, loc=0, scale=1),
                                   scipy.stats.lognorm(s=0.5, loc=0, scale=0.08)),
    "mig_rate_post_split": ScaledDist(scipy.stats.truncnorm(a=0, b=5, loc=0, scale=1),
                                      scipy.stats.truncnorm(a=0, b=5, loc=0, scale=0.2)),
    "mig_rate_wild": ScaledDist(scipy.stats.lognorm(s=0.5, loc=0, scale=1),
                                scipy.stats.lognorm(s=0.5, loc=0, scale=0.08)),
    "pop_size_wild_1": ScaledDist(scipy.stats.lognorm(s=0.5, loc=0, scale=1),
                                  scipy.stats.lognorm(s=0.5, loc=30, scale=300)),
    "pop_size_wild_2": ScaledDist(scipy.stats.lognorm(s=0.5, loc=0, scale=1),
                                  scipy.stats.lognorm(s=0.5, loc=30, scale=300)),
    "pop_size_captive": ScaledDist(scipy.stats.lognorm(s=0.5, loc=0, scale=1),
                                   scipy.stats.lognorm(s=0.5, loc=30, scale=100)),
    "pop_size_domestic_1": ScaledDist(scipy.stats.lognorm(s=0.5, loc=0, scale=1),
                                      scipy.stats.lognorm(s=0.5, loc=30, scale=300)),
    "pop_size_domestic_2": ScaledDist(scipy.stats.lognorm(s=0.5, loc=0, scale=1),
                                      scipy.stats.lognorm(s=0.5, loc=30, scale=300))
}


### Add priors to model and plot ###
m = elfi.ElfiModel("m")

elfi.Constant(int(5e6), name="length", model=m)
elfi.Constant(1.8e-8, name="recombination_rate", model=m)
elfi.Constant(6e-8, name="mutation_rate", model=m)

for prior_name, prior in priors.items():
    elfi.Prior(prior.sampling, name=prior_name, model=m)
    prior.plot(x_lab=prior_name)
    plt.savefig("../plots/prior/{}.png".format(prior_name))
    plt.clf()

y_obs = elfi_sim(
        bottleneck_strength_domestic=[3000],
        bottleneck_strength_wild=[30000],
        bottleneck_time_domestic=[3000],
        bottleneck_time_wild=[4000],
        captive_time=[20],
        div_time=[35000],
        mig_length_post_split=[1000],
        mig_length_wild=[20],
        mig_rate_captive=[0.01],
        mig_rate_post_split=[0.1],
        mig_rate_wild=[0.01],
        pop_size_captive=[100],
        pop_size_domestic_1=[200],
        pop_size_domestic_2=[200],
        pop_size_wild_1=[200],
        pop_size_wild_2=[200],
        length=int(10e6),
        recombination_rate=1.8e-8,
        mutation_rate=6e-8,
        random_state=np.random.RandomState(3),
        batch_size=1
)

### Set up other nodes ###
prior_args = [m[name] for name in m.parameter_names]  # Model only contains priors and constants

y = elfi.Simulator(elfi_sim, *prior_args, m["length"], m["recombination_rate"],
                   m["mutation_rate"], priors, name="simulator", observed=y_obs)

s = elfi.Summary(elfi_summary, y, None, True, name='s', model=m)  # None = no scaler, True = quick_mode

d = elfi.Distance('euclidean', s, name='d', model=m)

### Rejection to "train" sum stat scaler ###
pool = elfi.OutputPool(['s'])
rej = elfi.Rejection(m['d'], batch_size=4, seed=1, pool=pool, max_parallel_batches=64)
rej_res = rej.sample(50, quantile=1, bar=False)  # Accept all
store = pool.get_store('s')
sum_stats = np.array(list(store.values()))
sum_stats = sum_stats.reshape(-1, sum_stats.shape[2])  # Drop batches axis
scaler = StandardScaler()
scaler.fit(sum_stats)
m["s"].become(elfi.Summary(elfi_summary, y, scaler, True, name='s_scaled', model=m))  # Scaler and quick_mode

elapsed_time = time.time() - start_time
logging.info(f"Rej completed at {elapsed_time/60:.2f} minutes.")

### Run SMC ###
smc = elfi.SMC(m['d'], batch_size=5, seed=2, max_parallel_batches=64)
N = 500
schedule = [12, 11, 10, 9]
smc_res = smc.sample(N, schedule)
logging.info(smc_res.summary(all=True))

elapsed_time = time.time() - start_time
logging.info(f"SMC completed at {elapsed_time/60:.2f} minutes.")

### Collect and write out results ###
results = pd.DataFrame(smc_res.samples_array, columns=sorted(priors.keys()))

# scale back up results
for name, dist in priors.items():
    results[name] = dist.scale_up_samples(results[name])

results["weights"] = smc_res.weights

results.to_csv("../output/smc_posterior.csv", index=False)

# Write out pdf evaluations
df_list = []
for name, dist in priors.items():
    x_lims = dist.target.ppf([0.001, 0.999])
    x = np.linspace(x_lims[0], x_lims[1], 1000)
    y = dist.pdf(x)
    df = pd.DataFrame({"x": x, "parameter": name, "value": y})
    df_list.append(df)

pdf_df = pd.concat(df_list)
pdf_df.to_csv("../output/prior_pdf.csv", index=False)

elapsed_time = time.time() - start_time
logging.info(f"Job completed at {elapsed_time/60:.2f} minutes.")
