import joblib
import allel
import tsinfer
genotypes = allel.GenotypeArray(joblib.load("test_genotypes.joblib"))
pos = joblib.load("test_pos.joblib")
print(genotypes.shape)
