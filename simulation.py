# -*- coding: utf-8 -*-

import sys

import numpy as np
from scipy.stats import norm, poisson

import matplotlib.pyplot as plt

# model
size       = 100000 #100000
mean       = 100              # ground truth
candidates = [mean, 1, 5, 50, 200, 1000]
sigma      = 10


# Histogram

## graph
fig, axes = plt.subplots(3, 2, figsize=(10, 6), sharex=True, sharey=True)
nrow, ncol = axes.shape
size_bin=40

## under null hypothesis
df = np.random.normal(loc=mean, scale=sigma, size=size);
pval = norm.cdf(x=df, loc=mean, scale=sigma)
axes[0,0].set_title('Null Hypothesis mu=%d' % (mean))
axes[0,0].hist(x=pval, bins=size_bin)

## not follow null hypothesis
row=0; col=1; cnt=1;
for idx, candidate in enumerate(candidates):
    if idx == 0: continue

    df = np.random.poisson(lam=candidate, size=size)
    pval = poisson.cdf(df, candidate)
    axes[row, col].set_title('lambda=%d' % (candidate))
    axes[row, col].hist(x=pval, bins=size_bin)

    # udpate
    cnt += 1

    # row
    if cnt % 2 == 0: row += 1
    # col
    if cnt % 2 == 0: 
        col = 0
    else:
        col += 1

plt.savefig("Hist_p_values.png")
plt.close()


# QQ plot
## graph
fig, axes = plt.subplots(3, 2, figsize=(10, 6), sharex=True, sharey=True)
nrow, ncol = axes.shape
size_bin=40

## under null hypothesis
df = np.random.normal(loc=mean, scale=sigma, size=size);
pval = norm.cdf(x=df, loc=mean, scale=sigma)
counts, bin_edges = np.histogram(pval, normed=False)
cdf = np.cumsum(counts)
axes[0,0].set_title('Null Hypothesis mu=%d' % (mean))
axes[0,0].set_xlim(0,1)
axes[0,0].set_ylim(0,1)
axes[0,0].plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), color='black',  linestyle='dashed')
axes[0,0].plot(bin_edges[1:], cdf / float(cdf[-1]))

## not follow null hypothesis
row=0; col=1; cnt=1;
for idx, candidate in enumerate(candidates):
    if idx == 0: continue

    df = np.random.poisson(lam=candidate, size=size)
    pval = poisson.cdf(df, candidate)
    counts, bin_edges = np.histogram(pval, normed=False)
    cdf = np.cumsum(counts)
    axes[row, col].set_xlim(0,1)
    axes[row, col].set_ylim(0,1)
    axes[row, col].set_title('lambda=%d' % (candidate))
    axes[row, col].plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), color='black',  linestyle='dashed')
    axes[row, col].plot(bin_edges[1:], cdf / float(cdf[-1]))

    # udpate
    cnt += 1

    # row
    if cnt % 2 == 0: row += 1
    # col
    if cnt % 2 == 0: 
        col = 0
    else:
        col += 1

plt.savefig("QQ_p_values.png")
plt.close()

