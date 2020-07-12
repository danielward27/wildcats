from sim.model import WildcatSimulation, SeqFeatures
import sim.sum_stats as ss
import time
import pandas as pd
import numpy as np
import os

start_time = time.time()

array_id = int(os.environ['PBS_ARRAYID'])

runs_per_task = 200    # Jobs in an array are referred to as tasks
start_index = array_id*runs_per_task
end_index = start_index+runs_per_task

prior_df = pd.read_feather("../output/prior.feather")

float_col_names = ["mig_rate_wild", "mig_rate_post_split"]
int_col_names = [x for x in list(prior_df) if x not in float_col_names]
prior_df[int_col_names] = prior_df[int_col_names].astype(int)

# Run the model runs_per_task times
for i in range(start_index, end_index):
    params = prior_df.astype(object).iloc[i].to_dict()  # Row of parameters

    slim_parameters = {
        'pop_size_domestic_1': params["pop_size_domestic_1"],  # Population sizes are diploid.
        'pop_size_wild_1': params["pop_size_wild_1"],
        'pop_size_captive': params["pop_size_captive"],
        'mig_length_wild': params["mig_length_wild"],
        'mig_rate_wild': params["mig_rate_wild"],  # Rate of migration from domestic -> wildcats
        'captive_time': params["captive_time"],  # Time captive population established in SLiM
    }

    recapitate_parameters = {
        'pop_size_domestic_2': params["pop_size_domestic_2"],
        'pop_size_wild_2': params["pop_size_wild_2"],
        'div_time': params["div_time"],
        'mig_rate_post_split': params["mig_rate_post_split"],
        'mig_length_post_split': params["mig_length_post_split"],
        'bottleneck_time_wild': params["bottleneck_time_wild"],
        'bottleneck_strength_wild': params["bottleneck_strength_wild"],
        'bottleneck_time_domestic': params["bottleneck_time_domestic"],
        'bottleneck_strength_domestic': params["bottleneck_strength_domestic"],
    }

    # Run model
    seq_features = SeqFeatures(length=int(10e6), recombination_rate=1.8e-8, mutation_rate=6e-8)
    print("Warning: Simulating a 10 Mb region only!")
    sim = WildcatSimulation(seq_features=seq_features, random_seed=params["random_seed"])
    command = sim.slim_command(**slim_parameters)

    try:         # Just to make sure results end up being NA instead of breaking the run.
        decap_trees = sim.run_slim(command)
        demographic_events = sim.demographic_model(**recapitate_parameters)
        tree_seq = sim.recapitate(decap_trees, demographic_events)

    except Exception:
        print("The simulation failed to run on parameter index {}".format(i))
        continue  # If something goes wrong go to next iteration (row will be np.nan)

    # Take a sample of individuals
    samples = sim.sample_nodes(tree_seq, [5, 30, 10])
    tree_seq = tree_seq.simplify(samples=np.concatenate(samples))
    genotypes = ss.genotypes(tree_seq)
    pos = ss.positions(tree_seq)
    pop_list = ss.pop_list(tree_seq)
    samples = ss.sampled_nodes(tree_seq)

    # Calculate summary statistics
    def pca_pipeline(genotypes, pos, pop_list):
        genotypes, pos = ss.maf_filter(genotypes, pos)
        genotypes = genotypes.to_n_alt()  # 012 with ind as cols
        genotypes, pos = ss.ld_prune(genotypes, pos)
        pca_stats = ss.pca_stats(genotypes, pop_list)
        return pca_stats

    # Using a list to call function in for loop so we can use try/except (in case any functions fail)
    summary_functions = [
        ss.tskit_stats(tree_seq, samples),
        ss.afs_stats(tree_seq, samples),
        ss.r2_stats(tree_seq, samples, [0, 1e6, 2e6, 4e6], labels=["0_1Mb", "1_2Mb", "2_4MB"]),
        ss.roh_stats(genotypes, pos, pop_list, seq_features.length),
        pca_pipeline(genotypes, pos, pop_list),
    ]

    stats_dict = {"random_seed": sim.random_seed}  # Random seed acts as ID

    for func in summary_functions:
        try:
            stat = func
        except Exception:
            print("The function {} threw an error on parameter index {}".format(func.__name__, i))
            stat = {}
        stats_dict = {**stats_dict, **stat}

    if i == start_index:  # Saves writing out all the parameter names to initiate the df beforehand
        stats_df = pd.DataFrame(np.nan, index=range(start_index, end_index), columns=stats_dict.keys())
        stats_df.loc[i] = list(stats_dict.values())
    else:
        for col, value in stats_dict.items():
            stats_df.loc[i, col] = value

stats_df = stats_df.reset_index(drop=True)
output_filepath = "../output/summary_stats/summary_stats_{}.feather".format(array_id)
stats_df.to_feather(output_filepath)

print("Simulations completed in {:.2f} hours".format((time.time() - start_time)/60**2))
