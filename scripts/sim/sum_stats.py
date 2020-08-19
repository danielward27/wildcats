"""
The summary statistics here are calculated on the output of the simulations. The collate_results function gives
the data object needed for calculating summary statistics (has genotypes, positions, allele_counts etc.)
"""

from scipy.stats import iqr
import numpy as np
import pandas as pd
import allel
import sim.utils
import tskit
import logging


def binned_sfs_mean(ac, bin_no=5):
    """
    Caclulates the mean allele counts in bins across the site frequency spectrum.
    Ignores position 0 which corresponds to monomophic sites (which can be non-zero in sub populations).

    Arguments
    -------------
    ac: Allele counts (in format returned from scikit allel)
    bin_no: number of roughly equally spaced bins.

    Returns
    ----------
    dictionary with the bin range as the key and the

    """
    sfs = allel.sfs_folded(ac)[1:]  # Drop monomorphic sites
    if bin_no > len(sfs):
        raise ValueError("The number of bins cannot exceed the length of the site frequency spectrum.")

    split_sfs = np.array_split(sfs, bin_no)  # Splits roughly equal

    idx = 0
    stats = {}
    for array in split_sfs:
        bin_mean = array.mean()
        bin_label = f"{idx}_{idx + len(array)}"
        stats[bin_label] = bin_mean
        idx += len(array)

    return stats


def traditional_stats(data):
    """
    Caclulates lots of (mostly) traditional statistics,
    that are summaries of the site frequency spectrum.

    Arguments
    ---------
    data: Named tuple of results (made by collate_results function)

    Returns
    ---------
    Nested dictionary of statistics
    """
    pop_names = ["domestic", "wild", "captive", "all_pops"]

    stats = {
        "sfs_mean": {},
        "diversity": {},
        "wattersons_theta": {},
        "tajimas_d": {},
        "observed_heterozygosity": {},
        "expected_heterozygosity": {},
        "segregating_sites": {},
        "monomorphic_sites": {},
        "roh_mean": {},
        "roh_iqr": {},
        "r2": {},
        "f3": {},
        "divergence": {},
        "fst": {},
        "f2": {},
    }

    for pop in pop_names:
        # One way statistics
        stats["sfs_mean"][pop] = binned_sfs_mean(data.allele_counts[pop])
        stats["diversity"][pop] = allel.sequence_diversity(data.positions, data.allele_counts[pop])
        stats["wattersons_theta"][pop] = allel.watterson_theta(data.positions, data.allele_counts[pop])
        stats["tajimas_d"][pop] = allel.tajima_d(data.allele_counts[pop], data.positions)
        stats["observed_heterozygosity"][pop] = allel.heterozygosity_observed(data.genotypes[pop]).mean()
        stats["expected_heterozygosity"][pop] = allel.heterozygosity_expected(data.allele_counts[pop].to_frequencies(), ploidy=2).mean()
        stats["segregating_sites"] = data.allele_counts[pop].count_segregating()


        if pop is not "all_pops":  # all_pops has no monomorphic sites
            stats["monomorphic_sites"][pop] = data.allele_counts[pop].count_non_segregating()

            # Three way statistics
            other_pops = [pop_name for pop_name in pop_names if pop_name not in ["all_pops", pop]]
            t, b = allel.patterson_f3(data.allele_counts[pop],
                                      data.allele_counts[other_pops[0]],
                                      data.allele_counts[other_pops[1]])
            stats["f3"][pop] = np.sum(t) / np.sum(b)

    # Two way statistics
    for comparison in ["domestic_wild", "domestic_captive", "wild_captive"]:
        p = comparison.split("_")
        stats["divergence"][comparison] = allel.sequence_divergence(data.positions,
                                                                    data.allele_counts[p[0]],
                                                                    data.allele_counts[p[1]])

        num, den = allel.hudson_fst(data.allele_counts[p[0]], data.allele_counts[p[1]])
        stats["fst"][comparison] = np.sum(num) / np.sum(den)
        stats["f2"][comparison] = allel.patterson_f2(data.allele_counts[p[0]], data.allele_counts[p[1]]).mean()

    return stats


def ld_stats(data):
    """
    Calculates LD related statistics, ROH and R2
    :param data: Data named tuple created by collate data function
    :return: Nested dictionary of statistics
    """
    stats = {
        "roh": {},
        "r2": {},
    }

    pop_names = ["domestic", "wild", "captive", "all_pops"]

    for pop in pop_names:
        roh_ = roh(data.genotypes[pop], data.positions)
        stats["roh"][pop] = {"mean": roh_.mean(),
                             "iqr": iqr(roh_)}

        r2_ = binned_r2(data.genotypes[pop].to_n_alt(), data.positions, data.seq_length,
                        [0, 0.5e6, 1e6, 2e6, 4e6], ["0_0.5mb", "0.5_1e6", "1_2mb", "2_4mb"])
        stats["r2"][pop] = r2_.groupby("bins")["r2"].median().to_dict()

    return stats


def roh(genotypes, positions):
    """
    Calculates runs of homozygosity for a population. Only works for diploid individuals.

    Parameters
    ----------
    genotypes: scikit allel 3d genotypes array shape (variants, individuals, 2)
    positions: np.array of positions corresponding to axis 0 of genotypes (the variants)

    returns
    -----------
    np.array of runs of homozygosity

    """
    if genotypes.shape[2] is not 2:
        raise ValueError("genotypes.shape[2] should be 2 for diploid individuals")
    is_hetero = genotypes[:, :, 0] != genotypes[:, :, 1]
    df = pd.DataFrame(np.cumsum(is_hetero, axis=0))  # Increment roh ID with each heterozygote
    df["position"] = positions
    df = df.melt(id_vars="position", var_name="individual", value_name="roh_id")
    het_positions = df.groupby(['individual', 'roh_id'])["position"].min().reset_index()
    het_distance = het_positions.groupby(['individual'])['position'].diff()
    het_distance = het_distance.dropna().to_numpy(int)  # First ROH gives NA (as no previous mutation for comparison)
    return het_distance


def r2(x, y):
    """
    Calculates rogers huff r2 between a 1d array of a variants genotypes, and a set of other variants. Genotypes are provided
    in 012 format (ie. 0==homozygote reference, 1==heterozygote and 2==homozygote alternate.
    Note, the same result could be achieved with np.corrcoef(x, y)[1:, 0]**2 but this has much less overhead.
    :param x: Focal variant of shape (n_individuals,)
    :param y: Other variants of shape (n_variants, n_individuals)
    :return: Vector of r2 values of shape y.shape[0]
    """
    np_arr_2d = np.vstack((x, y))
    demeaned = np_arr_2d - np_arr_2d.mean(axis=1)[:, None]
    res = demeaned@demeaned[0]
    row_norms = np.sqrt((demeaned ** 2).sum(axis=1))
    res = res / (row_norms * row_norms[0])
    res = res**2
    return res[1:]


def binned_r2(genotypes, pos, seq_length, bins, labels,
              n_focal_muts=100, comparison_mut_lim=1000):
    """
    Caclulates rogers huff in different bin lengths across the genome. Random focal mutations are chosen,
    and R2 is calculated between each focal mutation to other mutations in the bins specified.
    :param genotypes: scikit allel 2d array of genotypes in 012 format (see allel to_n_alt() method)
    :param pos: ascending positions array corresponding to axis 0 of genotypes
    :param seq_length: int, Sequence length
    :param bins: list, list of bins e.g [0, 1e6, 2e6] would do 0-1Mb and 1-2Mb
    :param labels: list containing string labels for the bins
    :param n_focal_muts: Number of random focal mutations to use
    :param comparison_mut_lim: Limit the number of comparison mutations for each focal mutation (to limit memory usage)
    :return: pd.DataFrame containing r2 values
    """
    genotypes, pos = sim.utils.monomorphic_012_filter(genotypes, pos)

    max_bin = bins[-1]
    min_bin = bins[0]

    # Find max index to ensure "room" for mutations to compare
    max_idx = np.where(seq_length - max_bin < pos)[0].min()

    df_list = []
    for i in range(0, n_focal_muts):
        focal_mut_idx = np.random.randint(0, max_idx)
        focal_mut_pos = pos[focal_mut_idx]
        next_muts_idx = np.where(np.logical_and(pos > focal_mut_pos + min_bin,
                                                pos < focal_mut_pos + max_bin))[0]

        df_i = pd.DataFrame({
            "index": next_muts_idx,
            "pos": pos[next_muts_idx],
        })
        df_i["dist"] = df_i["pos"] - focal_mut_pos
        df_i["bins"] = pd.cut(df_i["dist"], bins, labels=labels)

        df_i = df_i.groupby("bins").apply(
            lambda x: x.sample(comparison_mut_lim) if len(x) > comparison_mut_lim else x).reset_index(drop=True)
        r2_ = r2(genotypes[focal_mut_idx], genotypes[df_i["index"]])
        df_i["r2"] = r2_
        df_list.append(df_i)

    results = pd.concat(df_list)
    return results


def pca(genotypes_012, subpops):
    """Carries out ld pruning and Patterson PCA of the genotypes.
    :param genotypes_012, genotype matrix in 012 (scikit-allel alt_n format)
    :param subpops, dictionary of subpopulation indexes
    :returns pd.DataFrame
    """
    genotypes_012 = sim.utils.monomorphic_012_filter(genotypes_012)
    genotypes_012 = sim.utils.ld_prune(genotypes_012)

    coords, model = allel.pca(genotypes_012, n_components=2, scaler='patterson')

    pca_data = pd.DataFrame({"pc1": coords[:, 0],
                             "pc2": coords[:, 1],
                             "population": ""})

    for pop in ["domestic", "wild", "captive"]:
        pca_data.loc[subpops[pop], "population"] = pop

    return pca_data


def pca_stats(pca_data):
    """
    Calculates the median and iqr of the populations, separately and combined, as well as the
    the pairwise distances between the medians.

    :param pca_data: pca data (produced by pca function)
    :return: nested dictionary of pca statistics for each population
    """
    # Pairwise median comparisons
    comparisons = [("domestic", "wild"), ("domestic", "captive"), ("wild", "captive")]
    medians = pca_data.groupby(["population"])[["pc1", "pc2"]].median()

    stats = {"pc1_median_dist": {}, "pc2_median_dist": {}}

    for comp in comparisons:
        for pc in ["pc1", "pc2"]:
            stats[f"{pc}_median_dist"][f"{comp[0]}_{comp[1]}"] = abs(medians[pc][comp[0]] - medians[pc][comp[1]])


    # Individual medians and iqr
    pca_data_all_pops = pca_data.copy()
    pca_data_all_pops["population"] = "all_pops"  # Append "all_pops" so we can groupby
    pca_data = pca_data.append(pca_data_all_pops)

    stats_df = pca_data.groupby("population").agg((np.median, iqr))
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns.values]
    stats = {**stats, **stats_df.to_dict()}

    return stats


def elfi_summary(data_array, scaler=None, quick_mode=False):
    """
    Calculates summary statistics on a numpy array of data with shape (batch_size, 1).
    :param data_array, np.array of GenotypeData objects
    :param scaler, scikit learn standard scaler to transform summary statistics
    :param quick_mode, bool, if True, only calculates traditional statistics, not PCA or LD ones.
    :returns np array of summary statistics with summary statistics as columns and rows as observations
    """
    results = []
    data_array = np.atleast_2d(data_array)
    data_array = data_array[:, 0]  # first col is everything needed (in a GenotypeData class)

    for data in data_array:
        data.allelify()  # Convert to scikit allel format

        trad_ss = traditional_stats(data)

        if quick_mode:
            logging.debug("Quick mode is activated")
            collated_ss = trad_ss
        else:
            pca_data = pca(data.genotypes["all_pops"].to_n_alt(), data.subpops)
            pca_ss = pca_stats(pca_data)
            ld_ss = ld_stats(data)
            collated_ss = {**trad_ss, **pca_ss, **ld_ss}

        collated_ss = sim.utils.flatten_dict(collated_ss)
        collated_ss = np.array(list(collated_ss.values()))
        results.append(collated_ss)

    results = np.array(results)
    if scaler is not None:
        results = scaler.transform(results)

    return results

