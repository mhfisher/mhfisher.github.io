---
layout: post
title:  "Tendency toward extrema from personalization [draft]"
date:   2022-09-17 00:24:34 -0400
categories: Notebooks
---
# Tendency toward extrema from personalization

## (Rough Draft) September 2022

This notebook presents a simple model of the tendency towards extreme drift from increasingly personalized interest groups. Consider a simple game with two objects, publications and readers. Readers have a preference scalar, and they naturally gravitate towards the publication that most closely matches their preference. The publications, to stay in business, are pulled toward their readers' preferences.

Let R be the number of readers in the game, and P the number of publications. We model gravity as a simple constant pull, rather than the inverse square law in physical gravity (for intuition, consider each reader an equally valuable customer of the publication). Let alpha be the force that each publication's readers exert on it, and beta be the force the each publication exerts on its readers.

## Case 1: Few publications

Consider first the case where only a few publications exist. A motivating case could be a print news market, where the fixed cost of starting a new publication is high. For now, we fix beta and alpha.


```python
from collections import defaultdict

import numpy as np
from scipy import stats

P = 2
R = 100
# Alpha and beta must be <= 1, else the desired property of pulling the publication
# and readers closer together is not achieved.
alpha = 0.1
beta = 0.1
timesteps = 1000
extreme_neg = -1
extreme_pos = 1

def _get_pub_cutoffs(P, pub_pref):
    """ Define the preference intervals within which readers will prefer each publication. """
    pub_cutoffs = []
    for pub_indx in range(P):
        left_next_pref = pub_pref[pub_indx - 1] if pub_indx > 0 else -float("inf")
        right_next_pref = pub_pref[pub_indx + 1] if pub_indx < (P - 1) else float("inf")
        left_cutoff = (pub_pref[pub_indx] + left_next_pref) / 2
        right_cutoff = (pub_pref[pub_indx] + right_next_pref) / 2
        pub_cutoffs.append((left_cutoff, right_cutoff))
    return pub_cutoffs
    
    
def simulate(P, R, beta, timesteps, extreme_neg, extreme_pos):
    # Initialize all readers to a normal distribution centered around 0
    readers = sorted(list(np.random.normal(size=100, scale=0.5)))
    # Initialize the publications equally dispersed between extrema.
    pub_step = abs(extreme_pos - extreme_neg) / (P + 1)
    pub_pref = [extreme_neg + pub_step * i for i in range(1, P+1)]
    for time in range(timesteps):
        pub_cutoffs = _get_pub_cutoffs(P, pub_pref)
        # Allocate readers to their current preferred publication
        # We store this mapping in both directions for convenience
        curr_pub_readers = defaultdict(list)
        curr_reader_prefs = {}
        for reader in readers:
            pub_indx = 0
            while not (pub_cutoffs[pub_indx][0] < reader < pub_cutoffs[pub_indx][1]):
                pub_indx += 1
            curr_pub_readers[pub_indx].append(reader)
            curr_reader_prefs[reader] = pub_indx
        # Readers exert a force alpha on publication
        for pub_indx, pub in enumerate(pub_pref):
            mean_reader = np.mean(curr_pub_readers[pub_indx])
            pub_pref[pub_indx] += (mean_reader - pub_pref[pub_indx]) * alpha
        # Publications exert a force beta on their readers
        for indx, reader in enumerate(readers):
            readers[indx] += (pub_pref[curr_reader_prefs[reader]] - reader) * beta
    print("Publications:")
    print(stats.describe(pub_pref))
    print("Readers:")
    print(stats.describe(readers))
    print("-------------")
    return readers, pub_pref
            

end_readers, end_publications = simulate(P, R, beta, timesteps, extreme_neg, extreme_pos)
```

    Publications:
    DescribeResult(nobs=2, minmax=(-0.3917613470428018, 0.3752443403702838), mean=-0.008258503336258993, variance=0.29414886226201, skewness=0.0, kurtosis=-2.0)
    Readers:
    DescribeResult(nobs=100, minmax=(-0.3917613470428021, 0.3752443403702841), mean=-0.04660878770691328, variance=0.14707443113100496, skewness=0.20100756305184247, kurtosis=-1.9595959595959591)
    -------------



```python
import matplotlib.pyplot as plt

plt.scatter(end_readers, [0]*len(end_readers))
plt.scatter(end_publications, [0]*len(end_publications), s=200, alpha=0.5)
plt.show()

```


![plot_P_2_stable](/assets/images/output_3_0.png)
    


As is intuitive, we see a convergence between the readers and the publications. One group of readers converges to the negative publication, and a second group converges to the positive one. The publications become slightly more extreme, but there is a centering effect on the readers, especially on those who start off more extreme.

## TODOs

1. Show a case with large P
2. Understand the numerical impacts of different values for alpha and beta.
3. Add a "sensationalism" coefficient to the publications, and show how for a large P, this causes a drift towards extrema for readers.
