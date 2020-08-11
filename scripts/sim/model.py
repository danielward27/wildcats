
"""
Script contains class that describes models of Scottish wildcat evolution. Scottish wildcats have
undergone extensive hybridisation with domestic cats. This model consists of a backwards in time coalescent simulation
with msprime (https://msprime.readthedocs.io/en/stable/)to estimate ancestral variation (ancvar) in wildcat and
domestic cats prior to hybridisation. A 500 generation forwards Wright-Fisher model can then be run from these two
ancestral populations using SLiM (https://messerlab.org/slim/), in which hybridisation occurs
between the two populations and a captive wildcat population is established.
"""

import msprime
import numpy as np
import pyslim
import subprocess
import os
from dataclasses import dataclass
import allel
from collections import namedtuple
import logging

@dataclass
class SeqFeatures:
    """Contains the fixed parameters in the model relating to the features of the sequence simulated"""
    length: int         # Length of sequence to simulate in base pairs.
    recombination_rate: float  # Recombination rate per base pair.
    mutation_rate: float     # Mutation rate per base pair.
    error_rate: float = 0  # Error rate (adds "mutations" per bp)


class WildcatSimulation:
    """Class outlines the model and parameters. Recommended that the directory "../output/" is set up so
    the default output paths work.

    Attributes:
        _pop_size_domestic_1, _pop_size_wild_1, _pop_size_captive (int): No. of diploid domestic, wild or captive cats.
        seq_features (object): instance of SeqFeatures dataclass
        random_seed (int): Random seed.
        suffix (bool): Adds _{random_seed} to filenames to avoid overwriting files.
        _decap_trees_filename (str): filename of the decapitated tree sequence outputted by SLiM.
    """

    def __init__(self, seq_features, random_seed=None, suffix=True):
        self.seq_features = seq_features
        self.random_seed = random_seed
        self.suffix = suffix
        self._decap_trees_filename = None
        self._pop_size_domestic_1 = None
        self._pop_size_wild_1 = None
        self._pop_size_captive = None

    def add_suffix(self, filename):
        """Helper function adds _{random_seed} before the last dot in certain filenames.
        Avoids clashes in filenames for example when running things in parallel."""
        dot_index = filename.rfind('.')
        filename = "{}_{}{}".format(filename[:dot_index], self.random_seed, filename[dot_index:])
        return filename

    def slim_command(self, param_dict,
                     decap_trees_filename="../output/decap.trees",
                     slim_script_filename='slim_model.slim'):
        """Uses a template slim command text file and replaces placeholders
        with parameter values to get a runnable command. Returns str.
                pop_size_domestic_1, pop_size_wild_1, pop_size_captive,
                     mig_rate_captive, mig_length_wild, mig_rate_wild, captive_time,
        TODO: correct description
        """
        param_dict2 = param_dict.copy()  # Avoid mutating original dictionary
        self._pop_size_domestic_1 = param_dict2["pop_size_domestic_1"]
        self._pop_size_wild_1 = param_dict2["pop_size_wild_1"]
        self._pop_size_captive = param_dict2["pop_size_captive"]

        if self.suffix:
            self._decap_trees_filename = self.add_suffix(decap_trees_filename)
        else:
            self._decap_trees_filename = decap_trees_filename

        param_dict2["length"] = self.seq_features.length
        param_dict2["recombination_rate"] = self.seq_features.recombination_rate
        command_no_params = 'slim {} -d decap_trees_filename=\'"{}"\' -s 40 {}'.format(
            {}, self._decap_trees_filename, slim_script_filename
        )

        params = ""
        for key, value in param_dict2.items():
            params += ("-d {}={} ".format(key, value))

        command = command_no_params.format(params)

        return command

    def run_slim(self, command):
        """Runs SLiM simulation from command line to get the decapitated tree sequence."""
        command_f = self.add_suffix("_temporary_command.txt")

        with open(command_f, 'w') as f:  # Running from file limits 'quoting games' (see SLiM manual pg. 425).
            f.write(command)
        try:
            logging.debug(command)  # Can set logging level to DEBUG to see
            subprocess.check_output(['bash', command_f], stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL)  # See https://bit.ly/3fMIWcE
            tree_seq = pyslim.load(self._decap_trees_filename)

        finally:  # Ensure files are cleaned up even if above fails
            os.remove(command_f)
            os.remove(self._decap_trees_filename)
        return tree_seq

    @staticmethod
    def demographic_model(pop_size_domestic_2, pop_size_wild_2, div_time, mig_rate_post_split,
                          mig_length_post_split, bottleneck_time_wild, bottleneck_strength_wild,
                          bottleneck_time_domestic, bottleneck_strength_domestic):
        """Model for recapitation, including bottlenecks, population size changes and migration.
        Returns list of demographic events sorted in time order. Note that if parameters are drawn from priors this
        could have unexpected consequences on the demography. sim.utils.test_prior() should mitigate this issue."""
        domestic, wild = 0, 1

        migration_time_2 = div_time-mig_length_post_split

        demographic_events = [
            msprime.PopulationParametersChange(time=bottleneck_time_domestic, initial_size=pop_size_domestic_2,
                                               population_id=domestic),  # pop size change executes "before" bottleneck
            msprime.InstantaneousBottleneck(time=bottleneck_time_domestic, strength=bottleneck_strength_domestic,
                                            population_id=domestic),
            msprime.PopulationParametersChange(time=bottleneck_time_wild, initial_size=pop_size_wild_2,
                                               population_id=wild),
            msprime.InstantaneousBottleneck(time=bottleneck_time_wild, strength=bottleneck_strength_wild,
                                            population_id=wild),
            msprime.MigrationRateChange(time=migration_time_2, rate=mig_rate_post_split, matrix_index=(domestic, wild)),
            msprime.MigrationRateChange(time=migration_time_2, rate=mig_rate_post_split, matrix_index=(wild, domestic)),
            msprime.MassMigration(time=div_time, source=domestic, dest=wild, proportion=1)]

        demographic_events.sort(key=lambda event: event.time, reverse=False)  # Ensure time sorted (required by msprime)
        return demographic_events

    def recapitate(self, decap_trees, demographic_events, demography_debugger=False):
        """Recapitates tree sequence under model specified by demographic events.
        Adds mutations and sequencing errors. Returns tskit.tree_sequence."""
        population_configurations = [
            msprime.PopulationConfiguration(initial_size=self._pop_size_domestic_1),  # msprime uses diploid Ne
            msprime.PopulationConfiguration(initial_size=self._pop_size_wild_1),
            msprime.PopulationConfiguration(initial_size=self._pop_size_captive)]

        tree_seq = decap_trees.recapitate(recombination_rate=self.seq_features.recombination_rate,
                                          population_configurations=population_configurations,
                                          demographic_events=demographic_events, random_seed=self.random_seed)

        # Overlay mutations
        tree_seq = pyslim.SlimTreeSequence(msprime.mutate(tree_seq, rate=self.seq_features.mutation_rate,
                                                          random_seed=self.random_seed))

        # Add sequencing errors
        if self.seq_features.error_rate != 0:
            tree_seq = pyslim.SlimTreeSequence(
                msprime.mutate(tree_seq, rate=self.seq_features.error_rate, random_seed=self.random_seed,
                               keep=True, start_time=0, end_time=1)
            )

        tree_seq = tree_seq.simplify()

        if demography_debugger:
            dd = msprime.DemographyDebugger(
                population_configurations=population_configurations,
                demographic_events=demographic_events)
            dd.print_history()

        return tree_seq

    def sample_nodes(self, tree_seq, sample_sizes, concatenate=True):
        """
        Samples nodes of individuals from the extant populations, which can then be used for simplification.
        Warning: Node IDs are not kept consistent during simplify!.

        Arguments
        ---------------
        tree_seq, tskit.tree_sequence: tree sequence object
        sample_sizes, list: a list of length 3, with each element giving the sample size
        for the domestic, wildcat and captive cat population respectively.
        concatenate, bool: If False, samples for each population are kept in seperate items in the list.

        Returns
        ------------
        np.array of nodes. If concatenate is false, an array for each population is returned in a list.

        """

        # Have to use individuals alive at to avoid remembered individuals at slim/msprime interface
        nodes = np.array([tree_seq.individual(i).nodes for i in tree_seq.individuals_alive_at(0)])
        pop = np.array([tree_seq.individual(i).population for i in tree_seq.individuals_alive_at(0)])
        np.random.seed(self.random_seed)
        samples = []
        for pop_num in range(0, 3):

            sampled_inds = np.random.choice(np.where(pop == pop_num)[0], replace=False,
                                            size=sample_sizes[pop_num])

            sampled_nodes = nodes[sampled_inds].ravel()
            samples.append(sampled_nodes)

        samples = np.concatenate(samples) if concatenate else samples
        return samples


def get_sampled_nodes(tree_seq):
    """
    Finds the sampled nodes from a simplified tree_sequence

    returns: namedtuple, where names are the population, and each tuple element is
    a numpy array of length 2 lists (the nodes from an individual)
    """
    Nodes = namedtuple("Nodes", "domestic, wild, captive")

    nodes = np.array([tree_seq.individual(i).nodes for i in tree_seq.individuals_alive_at(0)])
    pop = np.array([tree_seq.individual(i).population for i in tree_seq.individuals_alive_at(0)])

    node_tuple = Nodes(domestic=nodes[pop == 0], wild=nodes[pop == 1], captive=nodes[pop == 2])

    return node_tuple


def tree_summary(tree_seq):
    """Prints summary of a tree sequence"""
    tree_heights = []
    for tree in tree_seq.trees():
        for root in tree.roots:
            tree_heights.append(tree.time(root))
    print("Number of trees: {}".format(tree_seq.num_trees))
    print("Trees coalesced: {}".format(sum([t.num_roots == 1 for t in tree_seq.trees()])))
    print("Tree heights: max={}, min={}, median={}".format(max(tree_heights), min(tree_heights),
                                                           np.median(tree_heights)))
    print("Number of alive individuals: {}".format(len(tree_seq.individuals_alive_at(0))))
    print("Number of samples: {}".format(tree_seq.num_samples))
    print("Number of populations: {}".format(tree_seq.num_populations))
    print("Number of variants: {}".format(tree_seq.num_mutations))
    print("Sequence length: {}".format(tree_seq.sequence_length))


Data = namedtuple("Data", "genotypes, positions, subpops, allele_counts, seq_length")  # Define outside function so pickle works


class Results:
    """
        Collates results from the simulation in a format ideal for scikit-allel analysis.
        :param tree_seq:
        :return: Results object with attributes calculated
    """
    def __init__(self, tree_seq):
        self.genotypes = None
        self.positions = None
        self.subpops = None
        self.allele_counts = None
        self.seq_length = None
        self._initialize(tree_seq)

    def _initialize(self, tree_seq):
        pops = np.array([tree_seq.individual(i).population for i in tree_seq.individuals_alive_at(0)])
        all_pops_genotypes = genotypes(tree_seq)
        positions = np.array([variant.position for variant in tree_seq.variants()])

        subpops = {
            'domestic': np.where(pops == 0)[0],
            'wild': np.where(pops == 1)[0],
            'captive': np.where(pops == 2)[0],
            'all_pops': pops
        }

        allele_counts = all_pops_genotypes.count_alleles_subpops(subpops)

        # Numpyfy objects so pickle-able
        all_pops_genotypes = np.array(all_pops_genotypes)
        allele_counts = {key: np.array(value) for key, value in allele_counts.items()}

        genotypes_ = {
            "domestic": all_pops_genotypes[:, pops == 0, :],
            "wild": all_pops_genotypes[:, pops == 1, :],
            "captive": all_pops_genotypes[:, pops == 2, :],
            "all_pops": all_pops_genotypes,
        }

        self.genotypes = genotypes_
        self.positions = positions
        self.subpops = subpops
        self.allele_counts = allele_counts
        self.seq_length = int(tree_seq.get_sequence_length())

    def allelify(self):
        """
        Updates genotypes and allele counts array to scikit-allel wrappers
        """
        self.genotypes = {key: allel.GenotypeArray(value) for key, value in self.genotypes.items()}  # Numpy -> allel
        self.allele_counts = {key: allel.AlleleCountsArray(value) for key, value in self.allele_counts.items()}


def genotypes(tree_seq):
    """ Returns sampled genotypes in scikit allel genotypes format"""
    samples = get_sampled_nodes(tree_seq)
    samples = np.concatenate(samples).flatten()
    haplotype_array = np.empty((tree_seq.num_mutations, len(samples)), dtype=np.int8)
    for j, variant in enumerate(tree_seq.variants(samples=samples)):  # output order corresponds to samples
        haplotype_array[j, :] = variant.genotypes
    haplotype_array = allel.HaplotypeArray(haplotype_array)
    allel_genotypes = haplotype_array.to_genotypes(ploidy=2)
    return allel_genotypes




def elfi_sim(
        bottleneck_strength_domestic,
        bottleneck_strength_wild,
        bottleneck_time_domestic,
        bottleneck_time_wild,
        captive_time,
        div_time,
        mig_length_post_split,
        mig_length_wild,
        mig_rate_captive,
        mig_rate_post_split,
        mig_rate_wild,
        pop_size_captive,
        pop_size_domestic_1,
        pop_size_domestic_2,
        pop_size_wild_1,
        pop_size_wild_2,
        length, recombination_rate, mutation_rate,
        scale_dist_dict=None,
        random_state, batch_size,
):
    """
    Runs the simulation "vectorised" in a way that works with elfi.
    Lists of parameters should be of length batch_size. Length, recombination rate and mutation rate
    are assumed to be fixed.

    :param pop_size_domestic_1: list, domestic population size initially used in slim
    :param pop_size_wild_1: list, wild population size initially used in slim
    :param pop_size_captive: list, captive population size established at captive_time
    :param mig_rate_captive: list, migration rate into the captive population from the wild population
    :param mig_length_wild: list, length in generations from present that migration domestic -> wild starts
    :param mig_rate_wild: list, migration rate from domestic into captive population
    :param captive_time: list, Time in generations ago that the captive population is established
    :param pop_size_domestic_2: list, Ancient (pre-bottleneck) domestic population size
    :param pop_size_wild_2: list, Ancient (pre-bottleneck) wild population size
    :param div_time: list, Divergence time between domestic cats and wildcats (lybica and silvestris)
    :param mig_rate_post_split: list, Migration rate post divergence
    :param mig_length_post_split: list, Number of generations post split migration occurs for
    :param bottleneck_time_wild: list, Wild bottleneck (corresponding to migrating to Britain
    :param bottleneck_strength_wild: list, equivalent generations
    :param bottleneck_time_domestic:list, time bottleneck occurs
    :param bottleneck_strength_domestic: list, equivalent generations
    :param length: int, sequence length in bp
    :param recombination_rate: int, recombination rate per base pair
    :param mutation_rate: int, mutation rate per base pair
    :param scale_dist_dict: A dictionary of sim.utils.ScaledDists, which is used to scale the
           parameters up to the target distribution. keys should match corresponding parameter names.
    :param random_state: np.RandomState object
    :param batch_size: number to run in serial
    :return: np.array of Results classes
    """
    # Do not define locals above this if statement
    if scale_dist_dict is not None:
        # Scale up values
        kwargs = locals()
        priors["div_time"].scale_up_samples(1)  # This format should should work...
        pass



    data_list = []
    seeds = random_state.randint(1, 2 ** 31, batch_size)


    # Constant sequence features
    seq_features = SeqFeatures(length, recombination_rate, mutation_rate)

    # Run simulation with different param values
    for i in range(0, batch_size):
        sim = WildcatSimulation(seq_features=seq_features, random_seed=seeds[i])

        # run slim
        slim_param_dict = {
            "pop_size_domestic_1": int(pop_size_domestic_1[i]),
            "pop_size_wild_1": int(pop_size_wild_1[i]),
            "pop_size_captive": int(pop_size_captive[i]),
            "mig_rate_captive": mig_rate_captive[i],
            "mig_length_wild": int(mig_length_wild[i]),
            "mig_rate_wild": mig_rate_wild[i],
            "captive_time": int(captive_time[i])
        }

        command = sim.slim_command(slim_param_dict)
        decap_trees = sim.run_slim(command)

        # run msprime
        recapitate_parameters = {
            'pop_size_domestic_2': int(pop_size_domestic_2[i]),
            'pop_size_wild_2': int(pop_size_wild_2[i]),
            'bottleneck_time_wild': int(bottleneck_time_wild[i]),
            'bottleneck_strength_wild': int(bottleneck_strength_wild[i]),
            'bottleneck_time_domestic': int(bottleneck_time_domestic[i]),
            'bottleneck_strength_domestic': int(bottleneck_strength_domestic[i]),
            'mig_rate_post_split': mig_rate_post_split[i],
            'mig_length_post_split': int(mig_length_post_split[i]),
            'div_time': int(div_time[i]),
        }

        demographic_events = sim.demographic_model(**recapitate_parameters)
        tree_seq = sim.recapitate(decap_trees, demographic_events)

        # Take samples to match number of samples to the WGS data
        samples = sim.sample_nodes(tree_seq, [5, 30, 10])
        tree_seq = tree_seq.simplify(samples=samples)
        data = Results(tree_seq)

        data_list.append(data)

    data_array = np.atleast_2d(data_list).reshape(-1, 1)
    return data_array


