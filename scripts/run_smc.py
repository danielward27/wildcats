import logging
import numpy as np
import pickle
import elfi
from sim.model import elfi_sim
from sim.sum_stats import elfi_summary
from sklearn.preprocessing import StandardScaler
import time

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
elfi.set_client("ipyparallel", profile="pbs")  # Assumes ipython profile named pbs

start_time = time.time()

min_pop = 30  # Truncate population sizes to ensure enough to sample

# Read in priors and define model
with open("../output/priors.pkl", "rb") as f:  # Priors specified in priors.py
    priors = pickle.load(f)

m = elfi.ElfiModel("m")

elfi.Constant(int(5e6), name="length", model=m)
elfi.Constant(1.8e-8, name="recombination_rate", model=m)
elfi.Constant(6e-8, name="mutation_rate", model=m)

for prior_name, prior in priors.items():
    elfi.Prior(prior.sampling, name=prior_name, model=m)

# Pseudo observed. Replace with real data.
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

# Set up other nodes
prior_args = [m[name] for name in m.parameter_names]

y = elfi.Simulator(elfi_sim, *prior_args, m["length"], m["recombination_rate"],
                   m["mutation_rate"], priors, name="simulator", observed=y_obs)

s = elfi.Summary(elfi_summary, y, None, True, name='s', model=m)  # None = no scaler, True = quick_mode

d = elfi.Distance('euclidean', s, name='d', model=m)

# Rejection to "train" sum stat scaler
pool = elfi.OutputPool(['s'])
rej = elfi.Rejection(m['d'], batch_size=4, seed=1, pool=pool, max_parallel_batches=64)
rej_res = rej.sample(100, quantile=1, bar=False)  # Accept all
store = pool.get_store('s')
sum_stats = np.array(list(store.values()))
sum_stats = sum_stats.reshape(-1, sum_stats.shape[2])  # Drop batches axis
scaler = StandardScaler()
scaler.fit(sum_stats)
m["s"].become(elfi.Summary(elfi_summary, y, scaler, True, name='s_scaled', model=m))  # Scaler and quick_mode

elapsed_time = time.time() - start_time
logging.info(f"Rej completed at {elapsed_time/60:.2f} minutes.")

# Run SMC
smc = elfi.SMC(m['d'], batch_size=5, seed=2, max_parallel_batches=64)
N = 1000
schedule = [12, 11, 10, 9]
smc_res = smc.sample(N, schedule)

elapsed_time = time.time() - start_time
logging.info(f"SMC completed at {elapsed_time/60:.2f} minutes.")

# Save results
smc_res.save("../output/smc_posterior.pkl")

# TODO: remove below after sorting out plotting from R with reticulate
# Collect and write out results
# results = pd.DataFrame(smc_res.samples_array, columns=sorted(priors.keys()))

# scale back up results
# for name, dist in priors.items():
#    results[name] = dist.scale_up_samples(results[name])

# results["weights"] = smc_res.weights

# results.to_csv("../output/smc_posterior.csv", index=False)

# Write out pdf evaluations
# df_list = []
# for name, dist in priors.items():
#    x_lims = dist.target.ppf([0.001, 0.999])
#    x = np.linspace(x_lims[0], x_lims[1], 1000)
#    y = dist.pdf(x)
#    df = pd.DataFrame({"x": x, "parameter": name, "value": y})
#    df_list.append(df)

# pdf_df = pd.concat(df_list)
# pdf_df.to_csv("../output/prior_pdf.csv", index=False)

elapsed_time = time.time() - start_time
logging.info(f"Job completed at {elapsed_time/60:.2f} minutes.")
