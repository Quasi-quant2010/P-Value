# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import norm, poisson

import matplotlib.pyplot as plt

# model
size       = 100000
mean       = 100              # ground truth
candidates = [mean, 50, 200, 100000]
sigma      = 10


# graph
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
nrow, ncol = axes.shape
size_bin=40


# under null hypothesis
df = np.random.normal(loc=mean, scale=sigma, size=size);
pval = norm.cdf(x=df, loc=mean, scale=sigma)
axes[0,0].set_title('Null Hypothesis mu=%d' % (mean))
axes[0,0].hist(x=pval, bins=size_bin)

# not follow null hypothesis
row=0
for idx, candidate in enumerate(candidates):
    if idx == 0: continue

    if idx % 2: col=1
    else: col=0

    if idx >= 2: row=1

    df = np.random.poisson(lam=candidate, size=size)
    pval = poisson.cdf(df, candidate)
    axes[row, col].set_title('lambda=%d' % (candidate))
    axes[row, col].hist(x=pval, bins=size_bin)


plt.savefig("p_values.png")
plt.close()
