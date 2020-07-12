import joblib
import allel
import tsinfer
import numpy as np
import msprime

genotypes = allel.GenotypeArray(joblib.load("test_genotypes.joblib"))
pos = joblib.load("test_pos.joblib")
haplotypes = genotypes.to_haplotypes()

with tsinfer.SampleData(sequence_length=20000000) as sample_data:
    for i in range(0, len(pos)):
        sample_data.add_site(pos[i], haplotypes[i, :], np.random.choice(["A", "C", "T", "G"], 2, replace=False).tolist())

inferred_ts = tsinfer.infer(sample_data)
inferred_ts.dump("tree_seq.trees")
