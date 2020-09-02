import time
import logging
import numpy as np
import elfi
from sim.model import elfi_sim
from sim.sum_stats import elfi_sum, elfi_sum_scaler
import pickle
from sklearn.preprocessing import StandardScaler

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

seq_length = 44648284  # E3 length is 44648284
train_scaler_n_sim = 256
smc_n_samples = 500

logging.info(f"Seq length is set to {seq_length}")

num_cores = 0
timeout = time.time() + 600  # seconds

while num_cores is 0:
    elfi.set_client("ipyparallel", profile="pbs")  # Assumes ipython profile named pbs
    c = elfi.client.get_client()
    num_cores = c.num_cores
    if time.time() > timeout:
        raise TimeoutError("Could not find any cores after 600 seconds")
    time.sleep(10)

logging.info(f"Ipyparallel client started with {num_cores} cores.")

with open("../output/priors.pkl", "rb") as f:  # Priors specified in priors.py
    priors = pickle.load(f)

with open("../data/e3_phased.pkl", "rb") as f:
    y_obs = np.atleast_2d(pickle.load(f))

# Set up elfi model
m = elfi.ElfiModel("m", observed={"sim": y_obs})

elfi.Constant(seq_length, name="length", model=m)
elfi.Constant(1.8e-8, name="recombination_rate", model=m)
elfi.Constant(6e-8, name="mutation_rate", model=m)

for prior_name, prior in priors.items():
    elfi.Prior(prior.sampling, name=prior_name, model=m)

prior_args = [m[name] for name in m.parameter_names]

elfi.Simulator(elfi_sim, *prior_args, m["length"], m["recombination_rate"],
               m["mutation_rate"], priors, name="sim", model=m)

elfi.Summary(elfi_sum, m["sim"], name="sum", model=m)
elfi.Summary(elfi_sum_scaler, m["sum"], None, name="sum_scaler")  # Placeholder (no scaling yet)
elfi.Distance('euclidean', m["sum_scaler"], name='d', model=m)

# Rejection to "train" sum stat scaler
start_time = time.time()
pool = elfi.OutputPool(['sum'])
rej = elfi.Rejection(m['d'], batch_size=1, seed=1, pool=pool)
rej.sample(train_scaler_n_sim, quantile=1, bar=False)  # Accept all
sum_stats = pool.get_store('sum')
sum_stats = np.array(list(sum_stats.values()))
sum_stats = sum_stats.reshape(-1, sum_stats.shape[2])  # Drop batches axis
scaler = StandardScaler()
scaler.fit(sum_stats)
elapsed_time = time.time() - start_time
logging.info(f"{train_scaler_n_sim} simulations to train standard scaler completed in {elapsed_time/60:.2f} minutes")

m["sum_scaler"].become(elfi.Summary(elfi_sum_scaler, m["sum"], scaler, model=m))  # Now carries out scaling

# Run SMC
start_time = time.time()
smc = elfi.SMC(m['d'], batch_size=1, seed=2)
schedule = [17, 16, 15]
smc_res = smc.sample(smc_n_samples, schedule, bar=False)
elapsed_time = time.time() - start_time
logging.info(f"SMC completed at {elapsed_time/60:.2f} minutes.")

smc_res.save("../output/smc_posterior.pkl")  # Save results

logging.info(f"Shutting down cluster")
c = elfi.client.get_client().ipp_client
c.shutdown(hub=True)  # This does seem to throw an error in the controller, but things do shutdown
