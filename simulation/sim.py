import msprime
import tskit
import numpy as np
import pyslim
import subprocess
import os
from collections import namedtuple
import logging
import shlex
import pandas as pd
import random
import re
import pickle
from tabulate import tabulate
from priors import priors
import scipy.stats
import time
import utils


class WildcatModel:

    def __init__(
        self,
        seq_length: int,
        recombination_rate: float,
        mutation_rate: float,
        decap_trees_filename: str = "decap.trees",
        add_seed_suffix: bool = True,
        ):
        """Wildcat model object. Recommended that the directory "../output/" is set up so
        the default output paths work.

        Args:
            seq_length (int): Length of sequence to simulate in base pairs.
            recombination_rate (float): Recombination rate per base pair.
            mutation_rate (float): Mutation rate per base pair.
            random_seed (Optional[int], optional): Random seed. Defaults to None.
            decap_trees_filename (str): File to write out the decapitated tree sequence from slim.
            add_seed_suffix (bool): Whether to append _{seed} before file extension to limit risk of file name clashes.
        """
        self.seq_length = seq_length
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.add_seed_suffix = add_seed_suffix
        self.decap_trees_filename = decap_trees_filename


    def simulate(
        self,
        captive_time: int,
        div_time: int,
        div_time_dom: int,
        div_time_scot: int,
        mig_length_scot: int,
        mig_rate_scot: float,
        mig_rate_captive: float,
        pop_size_captive: int,
        pop_size_domestic_1: int,
        pop_size_scot_1: int,
        pop_size_eu_1: int,
        pop_size_lyb_1: int,
        pop_size_eu_2: int,
        pop_size_lyb_2: int,
        n_samples=[30, 30, 30, 30, 30],
        seed=None
        ):
        """

        Args:
            captive_time (int): Time captive population introduced (generations from present).
            div_time1 (int): Sylvestris-Lybica divergence time (generations from present).
            mig_length_post_split (int): Generations of symetric migration after Sylvestic-Lybica divergence.
            mig_rate_post_split (float): Strength of symetric migration after Sylvestic-Lybica divergence.
            mig_length_wild (int): Length of domestic to wild-living wildcat population migration (generations from present).
            mig_rate_wild (float): Rate of migration domestic -> wild-living.
            mig_rate_captive (float): Rate of migration of wildcats into captive wildcat population.
            pop_size_captive (int): Captive population size.
            pop_size_domestic_1 (int): Domestic population size (initially used in slim).
            pop_size_domestic_2 (int): Domestic population size pre-bottleneck.
            pop_size_wild_1 (int): Wild-living wildcat population size (initially used in slim).
            pop_size_wild_2 (int): Wild-living wildcat size pre-bottleneck.
            n_samples (list, optional): Number of samples from each population (domestic, wildcat, captive). Defaults to [5, 30, 10].
            seed (int, optional): Random seed. Defaults to None.
        """

        np.random.seed(seed)

        start = time.time() 

        # Run simulation
        command = self.slim_command(
            pop_size_domestic_1=int(pop_size_domestic_1),
            pop_size_scot_1=int(pop_size_scot_1),
            pop_size_eu_1=int(pop_size_eu_1),
            pop_size_lyb_1=int(pop_size_lyb_1),
            pop_size_captive=int(pop_size_captive),
            mig_rate_captive=mig_rate_captive,
            mig_length_scot=int(mig_length_scot),
            mig_rate_scot=mig_rate_scot,
            captive_time=int(captive_time),
            seed=seed
        )

        tree_seq = self.run_slim(command)  # Recent history (decapitated)


        demography = self.get_demography(
            pop_size_domestic_1=int(pop_size_domestic_1),
            pop_size_scot_1=int(pop_size_scot_1),
            pop_size_eu_1=int(pop_size_eu_1),
            pop_size_lyb_1=int(pop_size_lyb_1),
            pop_size_captive=int(pop_size_captive),
            pop_size_lyb_2=int(pop_size_lyb_2),
            pop_size_eu_2=int(pop_size_eu_2),
            div_time=int(div_time),
            div_time_dom=int(div_time_dom),
            div_time_scot=int(div_time_scot),
        )

        tree_seq = self.recapitate(tree_seq, demography, seed)


        # Take samples to match number of samples in real WGS data
        samples = self.sample_nodes(tree_seq, n_samples, seed)

        # simplify tree sequence
        tree_seq = tree_seq.simplify(samples=samples)

        # apply minor allele count filter to match real genome data
        tree_seq = tree_seq.delete_sites(utils.mac_filter(tree_seq, count=3))

        # apply thinning to match real genome data
        #tree_seq = tree_seq.delete_sites(utils.thinning(tree_seq, window=2000))

        end = time.time()
        print(tree_seq.num_sites)
        time_taken = end - start

        return tree_seq, time_taken

    def slim_command(
        self,
        pop_size_domestic_1,
        pop_size_lyb_1,
        pop_size_eu_1,
        pop_size_scot_1,
        pop_size_captive,
        mig_rate_captive,
        mig_length_scot,
        mig_rate_scot,
        captive_time,
        seed,
        slim_script_filename='slim_model.slim',
        add_seed_suffix: bool = True,
        ):

        param_dict = {
            "pop_size_domestic_1": pop_size_domestic_1,
            "pop_size_eu_1": pop_size_eu_1,
            "pop_size_scot_1": pop_size_scot_1,
            "pop_size_lyb_1": pop_size_lyb_1,
            "pop_size_captive": pop_size_captive,
            "mig_rate_captive": mig_rate_captive,
            "mig_length_scot": mig_length_scot,
            "mig_rate_scot": mig_rate_scot,
            "captive_time": captive_time,
            "length": self.seq_length,
            "recombination_rate": self.recombination_rate
        }

        if add_seed_suffix:
            decap_trees_filename = add_seed_suffix_to_file(self.decap_trees_filename, seed)

        param_string = "".join([f"-d {k}={v} " for k,v in param_dict.items()])
        command = f"""slim {param_string} -d decap_trees_filename="{decap_trees_filename}" -s {seed} {slim_script_filename} """

        return command

    def run_slim(self, command):
        """Runs SLiM simulation from command line to get the decapitated tree sequence."""

        command_split = shlex.split(command, posix=False)

        if self.add_seed_suffix:
            seed = re.findall(r"%s(\d+)" % " -s ", command)[0]  # regex to find seed
            decap_trees_filename = add_seed_suffix_to_file(self.decap_trees_filename, seed)
        else:
            decap_trees_filename = self.decap_trees_filenames
        
        try:
            logging.info(f"Running command: {command}")
            subprocess.check_output(command_split)  # This may not work on windows!
            tree_seq = tskit.load(decap_trees_filename)

        except subprocess.CalledProcessError as err:
            print(f"Running slim failed with command {command}")
            raise err

        finally:  # Ensure file are cleaned up even if above fails
            os.remove(decap_trees_filename)

        return tree_seq

    @staticmethod
    def get_demography(
        pop_size_domestic_1, pop_size_eu_1, pop_size_lyb_1, pop_size_scot_1, pop_size_captive, pop_size_eu_2,
        pop_size_lyb_2, div_time, div_time_scot, div_time_dom):
        """Model for recapitation, including bottlenecks, population size changes and migration.
        Returns list of demographic events sorted in time order. Note that if parameters are drawn from priors this
        could have unexpected consequences on the demography. sim.utils.check_params() should mitigate this issue."""


        domestic, scot, captive, eu, lyb = "p0", "p1", "p2", "p3", "p4"  # Names match slim for recapitation

        ## THESE MUST BE ADDED IN ORDER AS THEY ARE IN THE ABOVE LIST
        demography = msprime.Demography()
        demography.add_population(name=domestic, initial_size=pop_size_domestic_1)
        demography.add_population(name=scot, initial_size=pop_size_scot_1)
        demography.add_population(name=captive, initial_size=pop_size_captive, initially_active=False)
        demography.add_population(name=eu, initial_size=pop_size_eu_1)
        demography.add_population(name=lyb, initial_size=pop_size_lyb_1)
        demography.add_population(name="lyb2", initial_size=pop_size_lyb_2, initially_active=False)
        demography.add_population(name="eu2", initial_size=pop_size_eu_2, initially_active=False)
        demography.add_population(name="mrca", initial_size=pop_size_eu_2, initially_active=False)

        demography.add_population_split(time=div_time_dom, derived=[domestic, lyb], ancestral="lyb2")

        demography.add_population_split(time=div_time_scot, derived=[eu, scot], ancestral="eu2")

        demography.add_population_split(time=div_time, derived=["lyb2", "eu2"], ancestral="mrca")

        demography.sort_events()

        return demography

    def recapitate(self, decap_trees, demography, seed: int, demography_debugger=False):
        """Recapitates tree sequence under model specified by demography.
        Adds mutations and sequencing errors. Returns tskit.tree_sequence."""
        
        tree_seq = pyslim.recapitate(
            decap_trees,
            recombination_rate=self.recombination_rate,
            demography=demography,
            random_seed=seed)

        # Overlay mutations
        tree_seq = msprime.mutate(
            tree_seq,
            rate=self.mutation_rate,
            random_seed=seed
            )

        tree_seq = tree_seq.simplify()

        if demography_debugger:
            dd = demography.debug()
            print("### Here is msprime population info ###")
            dd.print_history()

        
        return tree_seq

    def sample_nodes(self, tree_seq, sample_sizes, seed, concatenate=True):
        """
        Samples nodes of individuals from the extant populations, which can then be used for simplification.
        Warning: Node IDs are not kept consistent during simplify!.

        param: tree_seq, tskit.tree_sequence: tree sequence object
        param: sample_sizes, list: a list of length 3, with each element giving the sample size
        for the domestic, wildcat and captive cat population respectively.
        param: concatenate, bool: If False, samples for each population are kept in seperate items in the list.

        Returns
        ------------
        np.array of nodes. If concatenate is false, an array for each population is returned in a list.

        """

        # Have to use individuals alive at to avoid remembered individuals at slim/msprime interface
        nodes = np.array([tree_seq.individual(i).nodes for i in pyslim.individuals_alive_at(tree_seq, 0)])
        pop = np.array([tree_seq.individual(i).population for i in pyslim.individuals_alive_at(tree_seq, 0)])
        np.random.seed(seed)
        samples = []
        for pop_num in range(5):
            sampled_inds = np.random.choice(
                np.where(pop == pop_num)[0], replace=False, size=sample_sizes[pop_num]
                )

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
    Nodes = namedtuple("Nodes", "domestic, scot, captive, eu, lyb")
    nodes = np.array([tree_seq.individual(i).nodes for i in pyslim.individuals_alive_at(tree_seq, 0)])
    pop = np.array([tree_seq.individual(i).population for i in pyslim.individuals_alive_at(tree_seq, 0)])
    node_tuple = Nodes(domestic=nodes[pop == 0], scot=nodes[pop == 1], captive=nodes[pop == 2], eu=nodes[pop == 3], lyb=nodes[pop == 4])

    return node_tuple


def add_seed_suffix_to_file(filename, seed):
    """Helper function adds _{random_seed} before the last dot in filenames. Avoids clashes in filenames."""
    dot_index = filename.rfind('.')
    filename = "{}_{}{}".format(filename[:dot_index], seed, filename[dot_index:])
    return filename


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
    print("Number of alive individuals: {}".format(len(pyslim.individuals_alive_at(tree_seq, 0))))
    print("Number of samples: {}".format(tree_seq.num_samples))
    print("Number of populations: {}".format(tree_seq.num_populations))
    print("Number of variants: {}".format(tree_seq.num_mutations))
    print("Sequence length: {}".format(tree_seq.sequence_length))


class GenotypeData:
    """
        Collates results from the simulation or real data in a format ideal for scikit-allel analysis.
        Genotype values greater than one when using a callset are set to 1 (i.e. assuming biallelic).
        :param tree_seq: tskit tree sequence, to generate results from simulations
        :param callset: scikit allel callset dictionary (from allel.read_vcf) for using real data
        :param subpops: dictionary of subpop names and corresponding indexes in genotypes when using callset
        :param seq_length: length of sequence when using callset
        :return: GenotypeData object with attributes calculated
    """
    def __init__(self, tree_seq=None, callset=None, subpops=None, seq_length=None):
        self.genotypes = None
        self.positions = None
        self.allele_counts = None
        self.subpops = subpops
        self.seq_length = seq_length

        if tree_seq is None and callset is None:
            raise ValueError("Either tree_seq or callset must be specified")

        if tree_seq is not None and callset is not None:
            raise ValueError("Only one of tree_seq or callset should be specified")

        if tree_seq is not None:
            self._initialize_from_tree_seq(tree_seq)

        if callset is not None:
            self._initialize_from_callset(callset)

    def _initialize_from_tree_seq(self, tree_seq):
        if self.subpops is not None:
            logging.warning("Ignoring subpops parameter, using info in tree_seq")
        if self.seq_length is not None:
            logging.warning("Ignoring seq_length parameter, using info in tree_seq")

