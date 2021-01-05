from sim.model import simple_sim
from sim.sum_stats import simple_sum
import logging
import time
import pandas as pd
import numpy as np
import os

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)

start_time = time.time()
array_id = int(os.environ['PBS_ARRAYID'])

runs_per_task = 200   # Jobs in an array are referred to as tasks
start_index = array_id*runs_per_task
end_index = start_index+runs_per_task

prior_df = pd.read_feather("../output/rejection/priors.feather")

failed_runs = 0

# Run the model runs_per_task times
for i in range(start_index, end_index):
    np.random.seed(i)
    params = prior_df.astype(object).iloc[i].to_dict()  # Row of parameters
    
    try:  # Just to make sure results end up being NA instead of breaking the run.
        data = simple_sim(**params, length=int(64340295),  # E2 length
                          recombination_rate=1.8e-8,
                          mutation_rate=6e-8, seed=i)  # Run model with params

    except Exception:
        logging.warning("The simulation failed to run on parameter index {}".format(i))
        failed_runs += 1
        if failed_runs == 2:
            raise ValueError('The number of failed runs in this batch reached 2!')
        continue  # If something goes wrong go to next iteration (row will be np.nan)

    stats_dict = simple_sum(data)
    stats_dict["random_seed"] = int(i)  # Add index/seed

    print("Calculating summary stats finished in {:.2f} s".format(time.time() - start_time))

    if i == start_index:  # Saves writing out all the parameter names to initiate the df beforehand
        stats_df = pd.DataFrame(np.nan, index=range(start_index, end_index), columns=stats_dict.keys())
        stats_df.loc[i] = list(stats_dict.values())
    else:
        for col, value in stats_dict.items():
            stats_df.loc[i, col] = value

stats_df = stats_df.reset_index(drop=True)
output_filepath = "../output/rejection/summary_stats/summary_stats_{}.feather".format(array_id)
stats_df.to_feather(output_filepath)
