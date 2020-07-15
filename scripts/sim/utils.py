import pandas as pd
import numpy as np
from pyarrow.lib import ArrowIOError

def merge_sum_stats(num_files):
    df_list = []
    missing_files = []

    for i in range(0, num_files):
        filename = "../output/summary_stats/summary_stats_{}.feather".format(i)
        try:
            df = pd.read_feather(filename)
            df = df.reset_index(drop=True)
            df_list.append(df)
        except ArrowIOError:
            missing_files.append(i)

    sum_stats = pd.concat(df_list, axis=0).reset_index(drop=True)
    sum_stats = sum_stats.sort_values(by="random_seed")

    # Check everything looks right
    seeds = sum_stats["random_seed"]

    difs = np.setdiff1d(np.arange(1,len(sum_stats)), np.array(seeds))

    if len(missing_files) != 0:
        print("Warning: there are {} files missing in specified range".format(len(missing_files)))
        if len(missing_files) < 10:
            print("The missing files are: {}".format(missing_files))

    if len(difs) != 0:
        print("Warning: np.setdiff1d suggests missing seeds")

    sum_stats.to_csv("../output/summary_stats.csv", index=False)


def test_prior(df):
    """Basic tests to unsure prior not malformed"""
    assert np.all(df >= 0), "Unexpected negative values in prior_df"

    # Check min pop_sizes are > sample sizes
    samp_size = {"pop_size_domestic_1": 5, "pop_size_wild_1": 30, "pop_size_captive": 10}

    for pop, samp_size in samp_size.items():
        assert np.all(df[pop] >= samp_size), "{} smaller than expected sample size {}".format(pop, samp_size)

    assert np.all(df[["captive_time", "mig_length_wild"]] < 500), "SLiM event scheduled > 500 generations ago"
    
    mig_rates = df[[col for col in list(df) if "mig_rate" in col]]
    assert np.all(mig_rates >= 0) & np.all(mig_rates <= 1)

    cond_1 = np.all(df[["div_time", "bottleneck_time_domestic", "bottleneck_time_wild"]] > 500)
    cond_2 = np.all(df["div_time"] - df["mig_length_post_split"] > 500)
    assert np.all([cond_1, cond_2]), "msprime event scheduled < 500 generations ago"

    cond_3 = np.all(df[["bottleneck_time_domestic", "bottleneck_time_wild"]].max(axis=1) < df["div_time"])
    assert cond_3, "Bottleneck scheduled to occur before divergence"

    print("Looks good!")

