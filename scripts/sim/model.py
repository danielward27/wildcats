
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
import pandas as pd
import pyslim
import subprocess
import os
from dataclasses import dataclass
import sim.sum_stats as ss
import sim.utils as utils

@dataclass
class SeqFeatures:
    """Contains the fixed parameters in the model relating to the features of the sequence simulated"""
    length: int         # Length of sequence to simulate in base pairs.
    recombination_rate: float  # Recombination rate per base pair.
    mutation_rate: float     # Mutation rate per base pair.
    error_rate: float = 0  # Error rate (adds "mutations" per bp)


class WildcatSimulation:
    """Class outlines the model and parameters. Recommended that the directory "../../output/" is set up so
    the default output paths work.

    Attributes:
        pop_size_domestic_1, pop_size_wild_1, pop_size_captive (int): Number of diploid domestic, wild or captive cats.
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

    def slim_command(self, pop_size_domestic_1, pop_size_wild_1, pop_size_captive,
                     mig_rate_captive, mig_length_wild, mig_rate_wild, captive_time,
                     decap_trees_filename="../output/decap.trees",
                     slim_script_filename='slim_model.slim',
                     template_filename='slim_command_template.txt'):
        """Uses a template slim command text file and replaces placeholders
        with parameter values to get a runnable command. Returns str."""

        self._pop_size_domestic_1 = pop_size_domestic_1
        self._pop_size_wild_1 = pop_size_wild_1
        self._pop_size_captive = pop_size_captive

        if self.suffix:
            self._decap_trees_filename = self.add_suffix(decap_trees_filename)
        else:
            self._decap_trees_filename = decap_trees_filename

        replacements_dict = {
            'p_pop_size_domestic_1': str(int(self._pop_size_domestic_1)),  # Placeholders prefixed with p_ in template
            'p_pop_size_wild_1': str(int(self._pop_size_wild_1)),
            'p_pop_size_captive': str(int(self._pop_size_captive)),
            'p_length': str(int(self.seq_features.length)),
            'p_recombination_rate': str(self.seq_features.recombination_rate),
            'p_mig_rate_captive': str(mig_rate_captive),
            'p_mig_length_wild': str(int(mig_length_wild)),
            'p_mig_rate_wild': str(mig_rate_wild),
            'p_captive_time': str(int(captive_time)),
            'p_random_seed': str(int(self.random_seed)),
            'p_slim_script_filename': slim_script_filename,
            'p_decap_trees_filename': self._decap_trees_filename,
        }

        with open(template_filename) as f:
            command = f.read()
            for placeholder, value in replacements_dict.items():
                if placeholder in command:
                    command = command.replace(placeholder, value)
                else:
                    print('Warning: the the placeholder {} could not be found in template file'.format(placeholder))
        return command

    def run_slim(self, command):
        """Runs SLiM simulation from command line to get the decapitated tree sequence."""
        command_f = self.add_suffix("_temporary_command.txt")

        with open(command_f, 'w') as f:  # Running from file limits 'quoting games' (see SLiM manual pg. 425).
            f.write(command)
        subprocess.run(['bash', command_f], stdout=subprocess.PIPE,  # See https://bit.ly/3fMIWcE
                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        tree_seq = pyslim.load(self._decap_trees_filename)
        os.remove(command_f)  # No need to keep the command file (can always print to standard out)
        os.remove(self._decap_trees_filename)  # We will delete the decapitated trees (don't need them).

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

    def sample_nodes(self, tree_seq, sample_sizes):
        """Get an initial sample of nodes from the populations that can be used for simplification.
        Sample_sizes provided in order [dom, wild, captive]. Note the node IDs are not consistent after
        simplification, although they can be accessed using the sampled_nodes function in SummaryStatistics class.
        """
        ind_df = individuals_df(tree_seq)
        sample_nodes = []
        for pop_num, samp_size in enumerate(sample_sizes):
            pop_df = ind_df[ind_df["population"] == pop_num]
            pop_sample = pop_df.sample(samp_size, random_state=self.random_seed)
            pop_sample_nodes = pop_sample["node_0"].tolist() + pop_sample["node_1"].tolist()
            sample_nodes.append(pop_sample_nodes)
        return np.array(sample_nodes)


def individuals_df(tree_seq):
    """Returns pd.DataFrame of individuals population and node indices."""
    individuals = tree_seq.individuals_alive_at(0)
    ind_dict = {
        "population": [],
        "node_0": [],
        "node_1": [],
    }
    for individual in individuals:
        ind = tree_seq.individual(individual)
        ind_dict["population"].append(ind.population)
        ind_dict["node_0"].append(ind.nodes[0])
        ind_dict["node_1"].append(ind.nodes[1])

    ind_df = pd.DataFrame(ind_dict)
    return ind_df


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


def run_sim(param_vec):
    """ I know this is a horrendous function but easyABC (R package) requires a function that does
    params -> summary stats, so that is what it does. Runs with no sequencing errors.
    parameters
    --------------
    param_vec: list of parameter values corresponding to the following parameter keys:
        ["random_seed", "pop_size_domestic_1", "pop_size_wild_1", "pop_size_captive", "captive_time",
         "mig_rate_captive", "mig_length_wild", "mig_rate_wild",  "pop_size_domestic_2",
         "pop_size_wild_2", "div_time", "mig_rate_post_split", "mig_length_post_split",
         "bottleneck_time_domestic", "bottleneck_strength_domestic", "seq_length",
         "bottleneck_time_wild", "bottleneck_strength_wild", "recombination_rate", "mutation_rate"]
    """

    param_keys = ["random_seed", "pop_size_domestic_1", "pop_size_wild_1", "pop_size_captive", "captive_time",
                  "mig_rate_captive", "mig_length_wild", "mig_rate_wild",  "pop_size_domestic_2",
                  "pop_size_wild_2", "div_time", "mig_rate_post_split", "mig_length_post_split",
                  "bottleneck_time_domestic", "bottleneck_strength_domestic", "bottleneck_time_wild",
                  "bottleneck_strength_wild", "seq_length", "recombination_rate", "mutation_rate"]
    params = dict(zip(param_keys, param_vec))

    int_param_names = [name for name in param_keys if "rate" not in name]
    for key, val in params.items():
        if key in int_param_names:
            params[key] = int(params[key])

    # Subset parameters based on function arguments
    slim_parameters = utils.get_params(params, WildcatSimulation.slim_command)
    recapitate_parameters = utils.get_params(params, WildcatSimulation.demographic_model)

    seq_features = SeqFeatures(params["seq_length"], params["recombination_rate"], params["mutation_rate"])
    sim = WildcatSimulation(seq_features=seq_features, random_seed=params["random_seed"])
    command = sim.slim_command(**slim_parameters)
    decap_trees = sim.run_slim(command)

    demographic_events = sim.demographic_model(**recapitate_parameters)
    tree_seq = sim.recapitate(decap_trees, demographic_events)

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
            print("The function {} threw an error".format(func.__name__))
            stat = {}
        stats_dict = {**stats_dict, **stat}

    print("The summary statistics calculated are:\n"
          "{}".format(stats_dict.keys()))

    return list(stats_dict.values())
