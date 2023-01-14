# Defines and plots priors using scipy.stats distributions

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sim.utils import plot_dist

priors = {
    "bottleneck_strength_domestic": scipy.stats.truncnorm(a=-0.5, b=np.inf, loc=7500, scale=15000),
    "bottleneck_strength_wild": scipy.stats.truncnorm(a=-0.5, b=np.inf, loc=7500, scale=15000),
    "bottleneck_time_domestic": scipy.stats.truncnorm(a=-6, b=np.inf, loc=3500, scale=500),
    "bottleneck_time_wild": scipy.stats.truncnorm(a=-6, b=np.inf, loc=3500, scale=500),
    "captive_time": scipy.stats.lognorm(s=0.4, loc=1, scale=np.exp(2.7)),
    "div_time": scipy.stats.truncnorm(a=-7, b=np.inf, loc=40000, scale=4000),
    "mig_length_post_split": scipy.stats.uniform(loc=0, scale=10000),
    "mig_length_wild": scipy.stats.lognorm(s=0.4, loc=1, scale=np.exp(2.5)),
    "mig_rate_captive": scipy.stats.lognorm(s=0.5, loc=0, scale=0.08),
    "mig_rate_post_split": scipy.stats.truncnorm(a=0, b=5, loc=0, scale=0.2),
    "mig_rate_wild": scipy.stats.lognorm(s=0.5, loc=0, scale=0.08),
    "pop_size_wild_1": scipy.stats.lognorm(s=0.2, loc=30, scale=np.exp(8.7)),
    "pop_size_wild_2": scipy.stats.lognorm(s=0.2, loc=30, scale=np.exp(9)),
    "pop_size_captive": scipy.stats.lognorm(s=0.5, loc=10, scale=100),
    "pop_size_domestic_1": scipy.stats.lognorm(s=0.25, loc=5, scale=np.exp(8.75)),
    "pop_size_domestic_2": scipy.stats.lognorm(s=0.2, loc=5, scale=np.exp(9.2))
}

def plot_priors(priors, **kwargs):
    
    fig, axes = plt.subplots(nrows=len(priors))

    for (name, dist), ax in zip(priors.items(), axes):
        x_min, x_max = dist.ppf([0.001, 0.999])
        x = np.linspace(x_min, x_max, 1000)
        y = dist.pdf(x)
        ax.plot(x, y, **kwargs)
        ax.set_xlabel(name)

    fig.set_size_inches(4, 20)
    fig.tight_layout()
    return fig, ax


fig, ax = plot_priors(priors)
fig.savefig("../plots/priors.png")

