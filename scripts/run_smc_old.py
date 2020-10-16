import logging
import numpy as np
import pickle
import elfi
from sim.model import elfi_sim
from sim.sum_stats import elfi_sum
from sklearn.preprocessing import StandardScaler
import time

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

seq_length = 44648284  # E3 length is 44648284
train_scaler_n_sim = 100
rej_n_sim = 100

logging.info(f"Seq length is set to {seq_length}")
logging.info(f"Will {train_scaler_n_sim} simulations to train scaler")
logging.info(f"Will use {rej_n_sim} simulations during rejection")

start_time = time.time()

# try:
#    num_cores = 0
#    timeout = time.time() + 600  # seconds
#    while num_cores is 0:
#        elfi.set_client("ipyparallel", profile="pbs")  # Assumes ipython profile named pbs
#        c = elfi.client.get_client()
#        num_cores = c.num_cores

#        if time.time() > timeout:
#            raise TimeoutError("Could not find any cores after 600 seconds")
#        time.sleep(10)

#    logging.info(f"Ipyparallel client started with {num_cores} cores.")


with open("../output/priors.pkl", "rb") as f:  # Priors specified in priors.py
    priors = pickle.load(f)

with open("../data/e3_phased.pkl", "rb") as f:  # Pickled GenotypeData object
    y_obs = np.atleast_2d(pickle.load(f))

m = elfi.ElfiModel("m")

elfi.Constant(seq_length, name="length", model=m)  # E3 size = 44648284
elfi.Constant(1.8e-8, name="recombination_rate", model=m)
elfi.Constant(6e-8, name="mutation_rate", model=m)

for prior_name, prior in priors.items():
    elfi.Prior(prior.sampling, name=prior_name, model=m)

prior_args = [m[name] for name in m.parameter_names]

y = elfi.Simulator(elfi_sim, *prior_args, m["length"], m["recombination_rate"],
                   m["mutation_rate"], priors, name="simulator", observed=y_obs)

s = elfi.Summary(elfi_sum, y, name='s', model=m)  # None = no scaler, False = all stats

d = elfi.Distance('euclidean', s, name='d', model=m)

# Rejection to "train" sum stat scaler
pool = elfi.OutputPool(['s'])
rej = elfi.Rejection(m['d'], batch_size=1, seed=1, pool=pool)
rej.sample(train_scaler_n_sim, quantile=1, bar=False)  # Accept all
store = pool.get_store('s')
sum_stats = np.array(list(store.values()))
sum_stats = sum_stats.reshape(-1, sum_stats.shape[2])  # Drop batches axis
scaler = StandardScaler()
scaler.fit(sum_stats)
m["s"].become(elfi.Summary(elfi_sum, y, scaler, True, name='s_scaled', model=m))  # Scaler and False=all stats

elapsed_time = time.time() - start_time
logging.info(f"Rej to train sum_stat scaler completed at {elapsed_time/60:.2f} minutes.")

# Using rejection just to get an idea of tolerances:
rej = elfi.Rejection(m['d'], batch_size=1, seed=1)
rej_res = rej.sample(rej_n_sim, quantile=1, bar=True)  # Accept all
np.save("../output/distances.npy", rej_res.discrepancies)

# TODO: Replace rejection above with SMC below once figured out ideal tolerances and increase rej sample size.
# Run SMC
#smc = elfi.SMC(m['d'], batch_size=1, seed=2, #max_parallel_batches=5)
#N = 5000
#schedule = [17, 15, 13, 12, 11, 10]
#smc_res = smc.sample(N, schedule, bar=False)

#elapsed_time = time.time() - start_time
#logging.info(f"SMC completed at {elapsed_time/60:.2f} minutes.")

# Save results
#smc_res.save("../output/smc_posterior.pkl")

elapsed_time = time.time() - start_time
logging.info(f"Job completed at {elapsed_time/60:.2f} minutes.")

# finally:
elapsed_time = time.time() - start_time
logging.info(f"Shutting down cluster at {elapsed_time/60:.2f} minutes")
c = elfi.client.get_client().ipp_client
c.shutdown(hub=True)  # This does seem to throw an error in the controller, but things do shutdown...
