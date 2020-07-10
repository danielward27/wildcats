# Imports
import scipy.stats as ss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sim.utils import test_prior

np.random.seed(2)

# Set up distributions
dist = {
    "pop_size_domestic_1": ss.uniform(100, 10000),  # ss.uniform(100, 15000),
    "pop_size_wild_1": ss.lognorm(0.4, scale=np.exp(8)),  # ss.lognorm(0.4, scale=np.exp(8.5)),
    "pop_size_captive": ss.lognorm(0.5, scale=np.exp(4.5)),
    "captive_time": ss.lognorm(0.7, scale=np.exp(3)),
    "migration_length_1": ss.lognorm(0.7, np.exp(3)),  # ss.uniform(0, 0) to remove
    "migration_rate_1": ss.beta(2, 50),  # ss.uniform(0, 0) to remove
    "bottleneck_strength_domestic": ss.uniform(0, 40000),
    "bottleneck_time_domestic": ss.norm(3500, 600),
    "pop_size_domestic_2": ss.uniform(1000, 20000),  # ss.uniform(1000, 25000),
    "bottleneck_strength_wild": ss.uniform(0, 40000),
    "bottleneck_time_wild": ss.norm(3500, 600),
    "pop_size_wild_2": ss.lognorm(0.2, scale=np.exp(8.8)),  #  ss.lognorm(0.2, scale=np.exp(9.2)),
    "migration_length_2": ss.uniform(0, 10000),
    "migration_rate_2": ss.truncnorm(0, 1, scale=0.03),
    "div_time": ss.norm(40000, 5000),
}

runs = 200000
df = pd.DataFrame({"random_seed": np.arange(1, runs + 1)})

for key, d in dist.items():
    # Sample from the prior
    df[key] = d.rvs(runs)

    # Plot PDFs
    x = np.linspace(d.ppf(1e-4), d.ppf(0.995), 1000)
    plt.figure(figsize=(15, 10))
    sns.lineplot(x, d.pdf(x), color="black")
    plt.subplots_adjust(left=0.1, bottom=0.1)
    plt.xlabel(key)
    plt.ylabel("Probability density")
    plt.savefig("../plots/prior_distributions/{}.png".format(key))
    plt.clf()

float_col_names = ["migration_rate_1", "migration_rate_2"]
int_col_names = [x for x in list(df) if x not in float_col_names]
df[int_col_names] = df[int_col_names].round().astype(int)

test_prior(df)

df.to_feather("../output/prior.feather")  # Binary df storage. Not designed for long term storage, but fast.
df.to_csv("../output/prior.csv", index=False)
