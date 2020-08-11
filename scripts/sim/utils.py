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


def maf_filter(allel_genotypes, pos=None, threshold=1, verbosity=0):
    """Remove minor alleles from genotypes and positions attributes.
    By default returns an scikit-allel 012 matrix (individuals as columns).

    Arguments
    -------------
    allel_genotypes: allel.GenotypeArray (3D)
    pos: positions array
    threshold: int, minor allele count threshold (default removes singletons)
    verbosity: int, If >0 prints how many variants retained
    """
    genotypes = allel_genotypes
    allele_counts = genotypes.count_alleles()
    maf_filter = allele_counts.min(axis=1) > threshold

    genotypes = genotypes.compress(maf_filter, axis=0)

    if verbosity > 0:
        print("maf_filter: Retaining: {}  out of {} variants".format(np.sum(maf_filter), len(maf_filter)))

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

def test_prior(df):
    """Basic tests to ensure prior not malformed"""
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


def test_adding_seq_errors():
    """A sanity check to make sure method of adding sequencing errors works as expected,
    i.e. mutations are added above only sample nodes in the tree sequence"""
    tree_seq = msprime.simulate(sample_size=50, Ne=1000, length=10e3, mutation_rate=0, random_seed=20,
                                recombination_rate=6e-8)
    # Add sequencing errors
    tree_seq = msprime.mutate(tree_seq, rate=1e-2, keep=True, end_time=1, start_time=0)
    for site in tree_seq.sites():
        for mutation in site.mutations:
            assert mutation.node in tree_seq.get_samples()
    print("{} mutations added only above sample nodes".format(tree_seq.num_mutations))
    print("Expected ~{} mutations".format(50*10e3*1e-2))


def get_params(dictionary, function):
    """Filters dictionary using argument names from a function"""
    return {key: param for key, param in dictionary.items()
            if key in function.__code__.co_varnames}


class ScaledDist:
    """Handles scaling of distributions for elfi (to ensure matrix is well-conditioned).
    Currently only supports scipy distributions with a loc and scale parameter.
    I have checked normal, uniform and lognormal work as intended.
    sampling: The frozen distribution that should be passed to the elfi prior
    target: The actual frozen distribution we want to be passed to the simulator"""

    def __init__(self, sampling, target):

        if type(sampling.dist) is not type(target.dist):
            raise ValueError("The sampling distribution and target distribution should be the same type.")

        if len(sampling.kwds) is 0 or len(target.kwds) is 0:
            raise ValueError("The parameters should be provided as keyword arguments.")

        if isinstance(sampling.dist, scipy.stats._continuous_distns.lognorm_gen):
            if sampling.kwds["s"] != target.kwds["s"]:
                raise ValueError("The shape parameters must match for scaling to function")

        self.sampling = sampling
        self.target = target

    def scale_up_samples(self, samples):
        """
        Scales rvs from sampling distribution to being samples the target distribution.
        """
        # Standardise samples
        samples = (samples - self.sampling.kwds["loc"]) / self.sampling.kwds["scale"]

        # Scale to target dist
        samples = samples * self.target.kwds["scale"] + self.target.kwds["loc"]
        return samples

    def plot(self, x_lab=""):
        """
        Plots disthandler object. Plots the target distribution against
        a kernal density estimate of a scaled up sample from the sampling distribution.
        Useful for checking scaling is functioning as expected.
        """
        sample = self.sampling.rvs(100000)
        scaled_sample = self.scale_up_samples(sample)

        plot_prior(self.target, x_lab, label="target")
        sns.kdeplot(scaled_sample, bw=0.1, label="scaled samples kde")
        return plt.plot()


def plot_prior(continous_dist, x_lab="", **kwargs):
    """
    Plots a pdf of a continuous varaiable.
    :param continous_dist: scipy.stats frozen continuous distribution
    :return: plot
    """
    x_lims = continous_dist.ppf([0.001, 0.999])
    x = np.linspace(x_lims[0], x_lims[1], 1000)
    y = continous_dist.pdf(x)
    plt.plot(x, y, "r-", **kwargs)
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.xlabel(x_lab)
    plt.ylabel("Probability density")

    return plt.plot()


