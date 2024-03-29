"""
Module contains assortment of functions for helping handle simulated data, files, tests etc.
"""

import pandas as pd
import numpy as np
import msprime
from pyarrow.lib import ArrowIOError
import allel
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import logging


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


def monomorphic_012_filter(genotypes_012, pos=None):
    """
    Removes variants from 012 format in which only a single state is observed.
    Note constant heterozygotes are also removed
    :param genotypes_012: 012 format of genotypes (see to_n_alt() of scikit allel).
    :param pos, positions vector
    :return: a 2 tuple of genotypes and positions if positions passed, else just genotypes
    """
    mono = np.any([np.all(genotypes_012 == 0, axis=1),
                   np.all(genotypes_012 == 1, axis=1),
                   np.all(genotypes_012 == 2, axis=1)], axis=0)

    if pos is not None:
        return genotypes_012[~mono], pos[~mono]
    else:
        return genotypes_012[~mono]


def maf_filter(genotypes, pos=None, threshold=1, verbosity=0):
    """Remove minor alleles from genotypes and positions attributes.
    By default returns an scikit-allel 012 matrix (individuals as columns).

    Arguments
    -------------
    genotypes: allel.GenotypeArray (3D)
    pos: positions array
    threshold: int, minor allele count threshold (default removes singletons)
    verbosity: int, If >0 prints how many variants retained
    """
    allele_counts = genotypes.count_alleles()
    maf_filter_ = allele_counts.min(axis=1) > threshold

    genotypes = genotypes.compress(maf_filter_, axis=0)

    if verbosity > 0:
        print("maf_filter: Retaining: {}  out of {} variants".format(np.sum(maf_filter_), len(maf_filter_)))

    if pos is not None:
        pos = pos[maf_filter]
        return genotypes, pos
    else:
        return genotypes


def ld_prune(genotypes_012, pos=None, size=100, step=20, threshold=0.1, verbosity=0):
    """Carries out ld pruning"""
    loc_unlinked = allel.locate_unlinked(genotypes_012, size=size,
                                         step=step, threshold=threshold)
    n = np.count_nonzero(loc_unlinked)
    genotypes = genotypes_012.compress(loc_unlinked, axis=0)
    if verbosity > 0:
        print("ld_prune: Retaining: {}  out of {} variants".format(n, genotypes_012.shape[0]))
    if pos is not None:
        pos = pos[loc_unlinked]
        return genotypes, pos
    else:
        return genotypes


def merge_sum_stats(num_files, filename, output_filename):
    """Merges 0 indexed incrementally numbered summary statistic feather files,
    into a single csv file.

    Parameters
    --------------
    num_files: number of files to merge
    filename: filepath to .feather files, number replaced with {}.
    output_filename: output csv filename.
    """
    df_list = []
    missing_files = []

    for i in range(0, num_files):
        file = filename.format(i)
        try:
            df = pd.read_feather(file)
            df = df.reset_index(drop=True)
            df_list.append(df)
        except ArrowIOError:
            missing_files.append(i)

    sum_stats = pd.concat(df_list, axis=0).reset_index(drop=True)
    sum_stats = sum_stats.sort_values(by="random_seed")

    # Check everything looks right
    seeds = sum_stats["random_seed"]
    difs = np.setdiff1d(np.arange(1, len(sum_stats)), np.array(seeds))

    if len(missing_files) != 0:
        print("Warning: there are {} files missing in specified range".format(len(missing_files)))
        if len(missing_files) < 10:
            print("The missing files are: {}".format(missing_files))

    if len(difs) != 0:
        print("Warning: np.setdiff1d suggests missing seeds")

    sum_stats.to_csv(output_filename, index=False)


# merge_sum_stats(500, "../../output/summary_stats/summary_stats_{}.feather", "../../output/summary_stats.csv")

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



