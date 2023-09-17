import cyvcf2
import tsinfer
import pickle
import builtins
import sys
import tskit
from tskit import MISSING_DATA
from samples_dict_2 import samples_dict

def add_diploid_sites(vcf, samples):
    """
    Read the sites in the vcf and add them to the samples object.
    """
    # You may want to change the following line, e.g. here we allow
    # "*" (a spanning deletion) to be a valid allele state
    allele_chars = set("ATGCatgc*")
    pos = 0
    for variant in vcf:  # Loop over variants, each assumed at a unique site
        if pos == variant.POS:
            print(f"Duplicate entries at position {pos}, ignoring all but the first")
            continue
        alleles = [variant.REF.upper()] + [v.upper() for v in variant.ALT]
        pos = variant.POS
        # Check we have ATCG alleles
        # Map original allele indexes to their indexes in the new alleles list.
        genotypes = [g for row in variant.genotypes for g in row[0:2]]
        samples.add_site(pos, genotypes, alleles)
    print("sites added")

def add_populations(vcf, samples):
    """
    Add tsinfer Population objects and returns a list of IDs corresponding to the VCF samples.
    """
    # In this VCF, the first letter of the sample name refers to the population
    #samples_first_letter = [sample_name[0] for sample_name in vcf.samples]
    pop_lookup = {}
    pop_lookup["domestic"] = samples.add_population(metadata={"pop": "domestic"})
    pop_lookup["scot"] = samples.add_population(metadata={"pop": "scot"})
    pop_lookup["captive"] = samples.add_population(metadata={"pop": "captive"})
    pop_lookup["eu"] = samples.add_population(metadata={"pop": "eu"})
    pop_lookup["lyb"] = samples.add_population(metadata={"pop": "lyb"})
    print("populations added")
    return [pop_lookup[population] for population in samples_dict.values()]


def add_diploid_individuals(vcf, samples, populations):
    for name, population in samples_dict.items():
        samples.add_individual(ploidy=2, metadata={"name": name}, population=population)
    print("individuals added")


def chromosome_length(vcf):
    return vcf.seqlens[0]


vcf_location = "./observed_01_nomiss.vcf.gz"
# NB: could also read from an online version by setting vcf_location to ./2022_BATCH2_E3.vcf.gz
# "https://github.com/tskit-dev/tsinfer/raw/main/docs/_static/P_dom_chr24_phased.vcf.gz"

vcf = cyvcf2.VCF(vcf_location)

#print("Chromosome length: ", chromosome_length(vcf))
with tsinfer.SampleData(
    path="E3.samples", sequence_length=44648254
) as samples:
    populations = add_populations(vcf, samples)
    add_diploid_individuals(vcf, samples, populations)
    add_diploid_sites(vcf, samples)


print(
    "Sample file created for {} samples ".format(samples.num_samples)
    + "({} individuals) ".format(samples.num_individuals)
    + "with {} variable sites.".format(samples.num_sites),
    flush=True,
)

# Do the inference
print("inferring tree")
ts = tsinfer.infer(samples, num_threads=8)
print(
    "Inferred tree sequence: {} trees over {} Mb ({} edges)".format(
        ts.num_trees, ts.sequence_length / 1e6, ts.num_edges
    )
)

with open("./tsinfer_tree.pickle", 'wb') as handle:
    pickle.dump(ts, handle, protocol=pickle.DEFAULT_PROTOCOL)