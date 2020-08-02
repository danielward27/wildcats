"""
The summary statistics here are calculated on the output of the simulations. The collate_results function gives
the data object needed for calculating summary statistics (has genotypes, positions, allele_counts etc.)
"""

from scipy.stats import iqr, pearsonr
import numpy as np
import pandas as pd
import allel
import tskit


def sfs_means(ac, bin_no=5):
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


def one_way_stats(data):
    """
    Caclulates diversity, tajimas d, wattersons theta, observed heterozygosity,
    and expected heterozygosity for the populations seperately and combined.

    Arguments
    ---------
    data: Named tuple of results (made by collate_results function)

    Returns
    ---------
    Nested dictionary of statistics
    """
    pop_names = ["domestic", "wild", "captive", "all_pops"]

    stats = {
        "sfs_means": {},
        "diversity": {},
        "wattersons_theta": {},
        "tajimas_d": {},
        "observed_heterozygosity": {},
        "expected_heterozygosity": {},
        "monomorphic_sites": {},
        "segregating_sites": {},
        "roh_mean": {},
        "roh_iqr": {},
    }

    for pop in pop_names:
        stats["sfs_means"][pop] = sfs_means(data.allele_counts[pop])
        stats["diversity"][pop] = allel.sequence_diversity(data.positions, data.allele_counts[pop])
        stats["wattersons_theta"][pop] = allel.watterson_theta(data.positions, data.allele_counts[pop])
        stats["tajimas_d"][pop] = allel.tajima_d(data.allele_counts[pop], data.positions)
        stats["observed_heterozygosity"][pop] = allel.heterozygosity_observed(data.genotypes[pop]).mean()
        stats["expected_heterozygosity"][pop] = allel.heterozygosity_expected(data.allele_counts[pop].to_frequencies(), ploidy=2).mean()
        stats["segregating_sites"] = data.allele_counts[pop].count_segregating()

        roh_ = roh(data.genotypes[pop], data.positions)
        stats["roh_mean"][pop] = roh_.mean()
        stats["roh_iqr"][pop] = iqr(roh_)

        if pop is not "all_pops":  # all_pops has no monomorphic sites
            stats["monomorphic_sites"][pop] = data.allele_counts[pop].count_non_segregating()

    return stats


def two_way_stats(data):
    """
    Calculates divergence, fst and f2 statistics for each population comparison.

    Arguments
    ---------
    data: Named tuple of results (made by collate_results function)

    Returns
    ---------
    Nested dictionary of statistics
    """
    stats = {
        "divergence": {},
        "fst": {},
        "f2": {},
    }

    for comparison in ["domestic_wild", "domestic_captive",
                       "wild_captive"]:  # Could be possible to add popA vs popB & popC
        p = comparison.split("_")
        stats["divergence"][comparison] = allel.sequence_divergence(data.positions,
                                                                    data.allele_counts[p[0]],
                                                                    data.allele_counts[p[1]])

        num, den = allel.hudson_fst(data.allele_counts[p[0]], data.allele_counts[p[1]])
        stats["fst"][comparison] = np.sum(num) / np.sum(den)
        stats["f2"][comparison] = allel.patterson_f2(data.allele_counts[p[0]], data.allele_counts[p[1]]).mean()

    return stats


def roh(genotypes, positions):
    """
    Calculates runs of homozygosity for a population. Only works for diploid individuals.

    Parameters
    ----------
    genotypes: scikit allel 3d genotypes array shape (variants, individuals, 2)
    positions: np.array of positions corresponding to axis 0 of genotypes

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

    # Filter monomorphic (or indeterminable due to 012)
    mono = np.any([np.all(genotypes == 0, axis=1),
                   np.all(genotypes == 1, axis=1),
                   np.all(genotypes == 2, axis=1)], axis=0)

    genotypes, pos = genotypes[~mono], pos[~mono]

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


def pca(genotypes_012, pop_list):
    """Patterson PCA of the genotypes. genotypes_012 is the scikit-allel 012 (alt_n) format. Returns df."""
    coords, model = allel.pca(genotypes_012, n_components=2, scaler='patterson')
    df = pd.DataFrame({"pc1": coords[:, 0],
                       "pc2": coords[:, 1],
                       "population": pop_list})
    df = df.melt(id_vars="population", var_name="pc")
    return df


def pca_iqr(df):
    """ Calculates the interquartile range of principle components using df outputted from pca function."""
    iqr_df = df.groupby(["population", "pc"])["value"].agg(iqr).reset_index()
    iqr_df["col_names"] = iqr_df["population"] + "_" + iqr_df["pc"] + "_iqr"
    iqr_dict = pd.Series(iqr_df["value"].values, index=iqr_df["col_names"]).to_dict()

    # Do for populations combined
    all_pops_iqr = df.groupby("pc")["value"].apply(iqr)
    iqr_dict["all_pops_pc1_iqr"] = all_pops_iqr["pc1"]
    iqr_dict["all_pops_pc2_iqr"] = all_pops_iqr["pc2"]
    return iqr_dict


def pca_pairwise_medians(df):
    """ Calculates the pairwise distances between the medians from the output of the pca function above."""
    comparisons = [("domestic", "wild"), ("domestic", "captive"), ("wild", "captive")]
    labels = ["dom_wild", "dom_cap", "wild_cap"]
    medians = df.groupby(["population", "pc"])["value"].agg("median").reset_index()
    pairwise_medians = pd.DataFrame()
    for comp, label in zip(comparisons, labels):
        comp_df = medians[medians["population"].isin(comp)].copy()
        dif = comp_df.groupby("pc")["value"].apply(lambda grp: abs(grp.min() - grp.max())).reset_index()
        dif["comparison"] = label
        pairwise_medians = pd.concat([pairwise_medians, dif])
    pairwise_medians["col_names"] = pairwise_medians["pc"] + "_pairwise_medians_" + pairwise_medians["comparison"]

    pairwise_medians_dict = pd.Series(pairwise_medians["value"].values,
                                      index=pairwise_medians["col_names"]).to_dict()
    return pairwise_medians_dict


def pca_stats(genotypes_012, pop_list):
    """Calculates pca summary stats (iqrs and pairwise distances)."""
    pca_df = pca(genotypes_012, pop_list)
    iqr_dict = pca_iqr(pca_df)
    medians_dict = pca_pairwise_medians(pca_df)
    pca_stats = {**iqr_dict, **medians_dict}
    return pca_stats


def collected_summaries(tree_seqs):
    results = []
    """
    Added for elfi
    # TODO: add documentation
    """

    # Calculate summary statistics
    def pca_pipeline(genotypes_, pos, pop_list):
        genotypes_, pos = maf_filter(genotypes_, pos)
        genotypes_ = genotypes_.to_n_alt()  # 012 with ind as cols
        genotypes_, pos = ld_prune(genotypes_, pos)
        pca_stats_dict = pca_stats(genotypes_, pop_list)
        return pca_stats_dict

    seq_length = int(tree_seqs[0].get_sequence_length())

    for tree_seq in tree_seqs:
        genotypes_ = genotypes(tree_seq)  # scikit-allel format
        pos = positions(tree_seq)
        nodes = sampled_nodes(tree_seq)
        pops = pop_list(tree_seq)

        # Using a list to call function in for loop so we can use try/except (in case any functions fail)
        summary_functions = [
            tskit_stats(tree_seq, nodes),
            afs_stats(tree_seq, nodes),
            # TODO: add these back in
            # r2_stats(tree_seq, sampled_nodes, [0, 1e6, 2e6, 4e6], ["0_1Mb", "1_2Mb", "2_4Mb"]),
            # roh_stats(genotypes_, pos, pops, seq_length),  # No longer works due to NetworkX 2.0 clash of requirements with elfi
            pca_pipeline(genotypes, pos, pop_list),
        ]

        # stats_dict = {"random_seed": sim.random_seed}  # Random seed acts as ID
        stats_dict = {}

        for func in summary_functions:
            stat = func
            stats_dict = {**stats_dict, **stat}

        results.append(list(stats_dict.values()))

    return np.array(results)
