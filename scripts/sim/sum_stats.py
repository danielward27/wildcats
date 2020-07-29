from scipy.stats import iqr, pearsonr
import numpy as np
import pandas as pd
import allel
import tskit
import sim.model



def sampled_nodes(tree_seq):
    """Gets a dictionary of all the nodes in the tree sequence. Individuals'
    nodes are kept adjacent, unlike the built in tree_seq.samples(population ={}).
    Returns: dict containing node indices for each population"""
    ind_df = sim.model.individuals_df(tree_seq)
    sampled_nodes = {
        "domestic": [],
        "wild": [],
        "captive": [],
        "all_pops": [],
    }
    for pop_num, (pop_name, node_list) in enumerate(sampled_nodes.items()):
        if pop_num == 3:
            pop_ind_df = ind_df  # All pops
        else:
            pop_ind_df = ind_df[ind_df["population"] == pop_num]
        pop_nodes_0 = np.array(pop_ind_df["node_0"])
        pop_nodes_1 = np.array(pop_ind_df["node_1"])
        pop_nodes = np.empty(2 * len(pop_nodes_0), dtype=int)
        pop_nodes[0::2] = pop_nodes_0  # Keeps individuals nodes adjacent in list
        pop_nodes[1::2] = pop_nodes_1
        sampled_nodes[pop_name] = list(pop_nodes)
    return sampled_nodes


def pop_list(tree_seq):
    """List of each pop name repeated by number of individuals in pop """
    sn = sampled_nodes(tree_seq)
    sn.pop("all_pops")
    pop_list_ = []
    for pop, nodes in sn.items():
        pop_list_ += int(len(nodes) / 2) * [pop]
    return pop_list_


def genotypes(tree_seq):
    """ Returns sampled genotypes in scikit allel genotypes format"""
    samples = sampled_nodes(tree_seq)["all_pops"]
    genotype_matrix = np.empty((tree_seq.num_mutations, len(samples)), dtype=np.int8)
    for j, variant in enumerate(tree_seq.variants(samples=samples)):  # output order corresponds to samples
        genotype_matrix[j, :] = variant.genotypes
    haplotype_array = allel.HaplotypeArray(genotype_matrix)
    allel_genotypes = haplotype_array.to_genotypes(ploidy=2)
    return allel_genotypes


def tskit_stats(tree_seq, sampled_nodes):
    """ Calculates the summary stats from tskit and returns them in a dictionary."""
    samples = [sampled_nodes["domestic"], sampled_nodes["wild"], sampled_nodes["captive"]]
    pop_names = ["domestic", "wild", "captive"]
    two_way_stats = ["fst", "f2", "divergence"]
    two_way_idxs = [(0, 1), (0, 2), (1, 2)]
    comparisons = ["dom_wild", "dom_cap", "wild_cap"]
    three_way_idxs = [(0, 1, 2), (1, 0, 2), (2, 0, 1)]

    stats = {"diversity": tree_seq.diversity(sample_sets=samples),
             "segregating_sites": tree_seq.segregating_sites(sample_sets=samples),
             "tajimas_d": tree_seq.Tajimas_D(sample_sets=samples),
             "f2": tree_seq.f2(sample_sets=samples, indexes=two_way_idxs),
             "divergence": tree_seq.divergence(sample_sets=samples, indexes=two_way_idxs),
             "f3": tree_seq.f3(sample_sets=samples, indexes=three_way_idxs),
             "y3": tree_seq.Y3(sample_sets=samples, indexes=three_way_idxs),
             "fst": tree_seq.Fst(sample_sets=samples, indexes=two_way_idxs),
             }

    tskit_stats = {}
    for stat, values in stats.items():
        for pop, comp, val in zip(pop_names, comparisons, values):
            if stat in two_way_stats:
                prefix = comp
            else:
                prefix = pop
            sum_stat_name = prefix + "_" + stat
            tskit_stats[sum_stat_name] = val

    # Do single way stats on whole population too
    samples = sampled_nodes["all_pops"]
    tskit_stats["all_pops_diversity"] = float(tree_seq.diversity(sample_sets=samples))
    tskit_stats["all_pops_segregating_sites"] = float(tree_seq.segregating_sites(sample_sets=samples))
    tskit_stats["all_pops_tajimas_d"] = float(tree_seq.Tajimas_D(sample_sets=samples))
    return tskit_stats


def afs_stats(tree_seq, sampled_nodes, bin_no=4):
    """Calculates bins of the allele frequency spectrum for each population,
    and the populations combined. Returns a dictionary.

    Arguments
    -----------
    tree_seq: tskit.TreeSequence
    sampled_nodes: dictionary of populations and associated sampled nodes list
    bin_no: Number of bins. Bins are then made of (approximately) constant size over frequency spectrum.
    """
    afs_stats = {}
    for pop, samples in sampled_nodes.items():
        afs = tree_seq.allele_frequency_spectrum(sample_sets=[samples], span_normalise=False)

        # Drop index 0 (monomorphic sites) and trailing zeros (where "MAF" > 0.5)
        afs = afs[1:int(len(samples) / 2) + 1]

        if bin_no > len(afs):
            raise ValueError("bin_no is greater than len(afs), try decreasing bin_no.")

        split_afs = np.array_split(afs, bin_no)
        bin_means = [array.mean() for array in split_afs]

        labels = []
        idx = 1
        for array in split_afs:
            labels.append("{}_afs_mean_{}_{}".format(pop, idx, idx + len(array) - 1))
            idx += len(array)

        pop_stats = dict(zip(labels, bin_means))
        afs_stats = {**afs_stats, **pop_stats}

    return afs_stats


def r2_data(genotypes, pos, seq_length, bins, labels, n_focal_muts=100, n_iter_muts=500):
    """Takes a scikit.allel.GenotypeArray and returns a df of r2 values for different bins.
    See r2_stats for more info."""
    haplotypes = genotypes.to_haplotypes()
    iterate_length = bins[-1]
    df_list = []
    # Find max index to avoid choosing focal mutation at end of chromosome
    max_idx = np.where(pos > seq_length - iterate_length)[0].min()
    for i in range(0, n_focal_muts):
        focal_mut_idx = np.random.randint(0, max_idx)
        focal_mut_pos = pos[focal_mut_idx]

        next_muts_idx = np.where(np.logical_and(pos > focal_mut_pos, pos < focal_mut_pos + iterate_length))[0]


        df_i = pd.DataFrame({
            "index": next_muts_idx,
            "pos": pos[next_muts_idx],
        })
        df_i["dist"] = df_i["pos"] - focal_mut_pos
        df_i["bins"] = pd.cut(df_i["dist"], bins, labels=labels)

        df_i = df_i.groupby("bins").apply(
            lambda x: x.sample(n_iter_muts) if len(x) > n_iter_muts else x).reset_index(drop=True)

        df_i["r2"] = df_i["index"].apply(
            lambda x: pearsonr(haplotypes[focal_mut_idx], haplotypes[x])[0]**2)

        df_list.append(df_i)

    results = pd.concat(df_list)

    return results


def r2_stats(tree_seq, sampled_nodes, bins, labels, n_focal_muts=100, n_iter_muts=500, summarise=True):
    """
    Calculates the r2 ld statistic in bins distances.
    Chooses focal mutations, iterates over subsequent mutations calculating r2
    and adds the values to the appropriate bins.

    Note: if memory usage is high, could summarise within the loop (instead of appending to a df)
    Warning: small bin sizes may have smaller sample sizes (due to fewer mutations)

    Arguments
    ------------
    tree_seq: tskit.TreeSequence
    sampled_nodes: dictionary of populations and nodes
    bins: list of bins e.g. [0, 1e6, 2e6] would create bins 0-1Mb and 1-2Mb
    labels: List of labels corresponding to bins
    n_focal_muts: int, the number of random focal mutations to choose
    n_iter_muts: int, number of mutations in each bin length to draw
                (takes all mutations if n_iter_muts > number of mutations in bin group)
    summarise: If False, returns a df of r2_data for each sample sets

    returns dictionary, with r2 mean and iqr for each population, for each bin
    """
    seq_length = tree_seq.get_sequence_length()

    df_list = []
    for pop, samples in sampled_nodes.items():
        ts = tree_seq.simplify(samples=samples)
        pos = positions(ts)
        gen = genotypes(ts)

        # Filter monomorphic sites
        gen, pos = maf_filter(gen, pos, threshold=0)

        r2_df = r2_data(gen, pos, seq_length, bins,
                        labels, n_focal_muts, n_iter_muts)
        r2_df = r2_df.drop(columns=["pos"])
        r2_df["population"] = pop
        df_list.append(r2_df)
    df = pd.concat(df_list)

    if summarise is False:
        return df
    else:
        df = df.groupby(["population", "bins"])["r2"].agg(r2_median=np.median, r2_iqr=iqr).reset_index()
        df = df.melt(id_vars=["population", "bins"], var_name="stat")
        df["dict_names"] = df["population"] + "_" + df["stat"] + "_" + df["bins"].astype("O")
        stat_dict = df[["dict_names", "value"]].set_index('dict_names').to_dict()["value"]
        return stat_dict


def roh_data(allel_genotypes, pos, pop_list, seq_length):
    """Returns a tuple containing a df roh lengths and a dictionary containing roh % coverage"""
    roh_cov_dict = {
        "domestic": [],
        "wild": [],
        "captive": []
    }

    roh_lengths_df = pd.DataFrame()
    for i, population in enumerate(pop_list):
        gv = allel_genotypes[:, i]
        roh_lengths, roh_cov = allel.roh_poissonhmm(gv, pos, phet_roh=0.0001,
                                                    phet_nonroh=(0.0025, 0.01),
                                                    transition=0.001, window_size=1000,
                                                    is_accessible=None,
                                                    contig_size=seq_length)
        roh_lengths["population"] = population
        roh_lengths_df = roh_lengths_df.append(roh_lengths)
        roh_cov_dict[population].append(roh_cov)
    return roh_lengths_df, roh_cov_dict


def roh_length_stats(roh_df):
    """Calculates the roh_length iqr and medians for the populations separately and combined. Returns dictionary."""
    length_stats = {}
    for function in [np.median, iqr]:
        stats = roh_df.groupby("population")["length"].apply(function)
        stats.index = stats.index + "_roh_length_" + function.__name__
        stats = stats.to_dict()
        stats["all_pops_roh_length_" + function.__name__] = function(roh_df["length"])
        length_stats = {**length_stats, **stats}
    return length_stats


def roh_cov_stats(roh_dict):
    """Calculates roh coverage median and iqr for the populations separately and combined. Returns dictionary."""
    coverage_stats = {}
    roh_dict["all_pops"] = [item for sublist in roh_dict.values() for item in sublist]

    for pop, cov_list in roh_dict.items():
        for function in [np.median, iqr]:
            sum_stat_name = pop + "_roh_cov_" + function.__name__
            coverage_stats[sum_stat_name] = function(cov_list)
    return coverage_stats


def roh_stats(allel_genotypes, pos, pop_list, seq_length):
    """Calculates roh length and coverage stats using above functions. Returns a dictionary."""
    roh_df, roh_dict = roh_data(allel_genotypes, pos, pop_list, seq_length)
    roh_length_stats_ = roh_length_stats(roh_df)
    roh_cov_stats_ = roh_cov_stats(roh_dict)
    roh_stats = {**roh_length_stats_, **roh_cov_stats_}
    return roh_stats


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


def positions(tree_seq):
    pos = []
    for variant in tree_seq.variants():
        pos.append(variant.position)
    pos = np.array(pos)
    return pos


def maf_filter(allel_genotypes, pos, threshold=1, verbosity=0):
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
    pos = pos[maf_filter]

    if verbosity > 0:
        print("maf_filter: Retaining: {}  out of {} variants".format(np.sum(maf_filter), len(maf_filter)))
    return genotypes, pos


def ld_prune(genotypes_012, pos, size=100, step=20, threshold=0.1, verbosity=0):
    """Carries out ld pruning"""
    loc_unlinked = allel.locate_unlinked(genotypes_012, size=size,
                                         step=step, threshold=threshold)
    n = np.count_nonzero(loc_unlinked)
    genotypes = genotypes_012.compress(loc_unlinked, axis=0)
    pos = pos[loc_unlinked]
    if verbosity > 0:
        print("ld_prune: Retaining: {}  out of {} variants".format(n, genotypes_012.shape[0]))
    return genotypes, pos


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
