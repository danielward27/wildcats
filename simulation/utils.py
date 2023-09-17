"""
Module contains assortment of functions for helping handle simulated data, files, tests etc.
"""

import pandas as pd
import numpy as np
import msprime
from pyarrow.lib import ArrowIOError
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import tskit
import random


def flatten_dict(d, sep='_'):
    """
    Recursively flattens a nested dictionary, concatenating the outer and inner keys.

    Arguments
    -----------
    stats_dict: A nested dictionary of statistics
    sep: seperator for keys

    Returns
    ------------
    dict
    """

    def items():
        for key, value in d.items():
            if isinstance(value, dict):
                for subkey, subvalue in flatten_dict(value).items():
                    yield key + sep + subkey, subvalue
            else:
                yield key, value

    return dict(items())


def check_params(df, n_samples):
    """Basic tests to ensure prior not malformed"""
    if np.any(df < 0):
        bad_params = df.columns[np.any(df <= 0, axis=0)]
        raise ValueError(f"Unexpected negative values at parameters {list(bad_params)}")

    # Check min pop_sizes are > sample sizes
    samp_size = {"pop_size_domestic_1": n_samples[0], "pop_size_wild_1": n_samples[1], "pop_size_captive": n_samples[2]}
    for pop, samp_size in samp_size.items():
        if np.any(df[pop] < samp_size):
            raise ValueError(f"{pop} smaller than expected sample size {samp_size}")

    for slim_gen_param in ["captive_time", "mig_length_wild"]:
        if np.any(df[slim_gen_param] > 500):
            raise ValueError(f"SLiM event scheduled > 500 generations ago for parameter {slim_gen_param}")

    for msprime_gen_param in ["div_time", "bottleneck_time_domestic", "bottleneck_time_wild"]:
        if np.any(df[msprime_gen_param] < 500):
            print(df[msprime_gen_param])
            raise ValueError(f"Parameter {msprime_gen_param} had values less than 500")

    if np.any(df["div_time"] - df["mig_length_post_split"] < 500):
        raise ValueError("div_time - mig_length_post_split was less than 500")

    mig_rate_params = [col for col in list(df) if "mig_rate" in col]
    if np.any(df[mig_rate_params] > 1) | np.any(df[mig_rate_params] < 0):
        raise ValueError("Migration rate parameter does not fall between 0 and 1")

    for bottleneck_param in ["bottleneck_time_domestic", "bottleneck_time_wild"]:
        if np.any(df[bottleneck_param] > df["div_time"]):
            raise ValueError(f"{bottleneck_param} scheduled to occur before div_time")


def test_adding_seq_errors():
    """A sanity check to make sure method of adding sequencing errors works as expected,
    i.e. mutations are added above only sample nodes in the tree sequence. This probably won't be used..."""
    tree_seq = msprime.simulate(sample_size=50, Ne=1000, length=10e3, mutation_rate=0, random_seed=20,
                                recombination_rate=6e-8)
    # Add sequencing errors
    tree_seq = msprime.mutate(tree_seq, rate=1e-2, keep=True, end_time=1, start_time=0)
    for site in tree_seq.sites():
        for mutation in site.mutations:
            assert mutation.node in tree_seq.get_samples()
    print("{} mutations added only above sample nodes".format(tree_seq.num_mutations))
    print("Expected ~{} mutations".format(50 * 10e3 * 1e-2))


def get_params(dictionary, function):
    """Filters dictionary using argument names from a function"""
    return {key: param for key, param in dictionary.items()
            if key in function.__code__.co_varnames}


def plot_dist(continuous_dist, x_lab="", **kwargs):
    """
    Plots a pdf of a continuous varaiable.
    :param continuous_dist: scipy.stats frozen continuous distribution
    :param x_lab, x-label string
    :return: plot
    """
    x_min, x_max = continuous_dist.ppf([0.001, 0.999])
    x = np.linspace(x_min, x_max, 1000)
    y = continuous_dist.pdf(x)
    plt.plot(x, y, **kwargs)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.xlabel(x_lab)
    plt.ylabel("Probability density")
    return plt.plot()


def mac_filter(ts, count):
    '''
    Minor allele count filter for tree sequence.
    ts: tree sequence
    count: minimum minor allele count for site to be kept
    returns: list of site IDs to remove
    '''
    variant = tskit.Variant(ts)
    mac_remove = []

    for site_id in range(ts.num_sites):
        variant.decode(site_id)
        if sum(variant.genotypes) not in range(count,(len(variant.genotypes)-count)):
            mac_remove.append(site_id)

    return mac_remove


def thinning(ts, window):
    '''
    Carries out thinning of sites, keeping one site per window size.
    ts: tree sequence
    window: window size
    returns: list of site IDs to remove
    '''
    window = 2000
    ids = []
    batch = []
    positions = zip(range(1,ts.num_sites),ts.tables.sites.position)
    for position in positions:
        if position[1] < window:
            batch.append(position[0])
        elif len(batch) == 0:
            window = window+window
        else:
            window = window+window
            ids.append(random.choice(batch))
            batch = []
    return list(set(range(0,ts.num_sites))-set(ids))

