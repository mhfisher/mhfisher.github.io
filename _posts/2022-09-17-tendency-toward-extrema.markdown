---
layout: post
title:  Tendency toward extrema from personalization [draft]
date:   2022-09-17 12:00:00 -0400
categories: Notebooks
classes: wide
---
## (Rough Draft) September 2022

This notebook presents a simple model of the tendency to drift towards extrema from increasingly personalized interest groups. Consider a simple game with two player types, publications and readers. Readers have a preference scalar, and they naturally gravitate towards the publication that most closely matches their preference. The publications, to stay in business, are pulled toward their readers' preferences.

Let R be the number of readers in the game, and P the number of publications. We model gravity as a simple constant pull, rather than the inverse square law in physical gravity (for intuition, consider each reader an equally valuable customer of the publication). Let alpha be the force that each publication's readers exert on it, and beta be the force that each publication exerts on its readers.

## Case 1: Few publications

Consider first the case where only a few publications exist. A motivating case could be a print news market, where the fixed cost of starting a new publication is high. For now, we fix beta equal to alpha.


```python
from collections import Counter, defaultdict

import numpy as np
import pprint
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

import matplotlib.pyplot as plt

def _plot(readers, publications):
    plt.scatter(readers, [0]*len(readers))
    plt.scatter(publications, [0]*len(publications), s=200, alpha=0.5)
    plt.show()

def _get_mean_preference_distance(readers):
    """ 
    Return the mean of preference differences across the readers, defined as:
    sum(|p_i - p_j| for all i != j) / |i != j|
    """
    R = len(readers)
    running_mean = 0
    known_distances = {}
    for i in range(len(readers)):
        reader_total_distance = 0
        for j in range(len(readers)):
            if i != j:
                if (i, j) not in known_distances:
                    known_distances[(i, j)] = abs(readers[i] - readers[j])
                reader_total_distance += known_distances[(i, j)]
        running_mean += reader_total_distance / (R - 1)
    return running_mean / R
            
    
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

def _print_stats(pub_prefs, readers, curr_reader_prefs=None):
    """ Print a header showing key simulation statistics """
    stats_header = "---------------\n"
    stats_header += f"Publication statistics:\n {stats.describe(pub_prefs)}\n"
    stats_header += f"Reader statistics:\n {stats.describe(readers)}\n"
    stats_header += f"Mean reader preference distance:\n {_get_mean_preference_distance(readers)}\n"
    if curr_reader_prefs:
        pub_reader_counts = defaultdict(int)
        for reader in readers:
            preferred_pub = curr_reader_prefs[reader]
            pub_reader_counts[preferred_pub] += 1
        stats_header += f"Publication reader counts:\n {dict(pub_reader_counts)}\n"
    stats_header += "---------------"
    print(stats_header)
    
    
def simulate(P, R, beta, timesteps, extreme_neg, extreme_pos):
    # Initialize all readers to a normal distribution centered around 0
    readers = sorted(list(np.random.normal(size=100, scale=0.5)))
    # Initialize the publications equally dispersed between extrema.
    pub_step = abs(extreme_pos - extreme_neg) / (P + 1)
    pub_prefs = [extreme_neg + pub_step * i for i in range(1, P+1)]
    _print_stats(readers, pub_prefs)
    _plot(readers, pub_prefs)
    for time in range(timesteps):
        pub_cutoffs = _get_pub_cutoffs(P, pub_prefs)
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
        for pub_indx, pub in enumerate(pub_prefs):
            mean_reader = np.mean(curr_pub_readers[pub_indx])
            pub_prefs[pub_indx] += (mean_reader - pub_prefs[pub_indx]) * alpha
        # Publications exert a force beta on their readers
        for indx, reader in enumerate(readers):
            readers[indx] += (pub_prefs[curr_reader_prefs[reader]] - reader) * beta
    _print_stats(pub_prefs, readers, curr_reader_prefs)
    return readers, pub_prefs
            

end_readers, end_publications = simulate(P, R, beta, timesteps, extreme_neg, extreme_pos)
_plot(end_readers, end_publications)
```

    ---------------
    Publication statistics:
     DescribeResult(nobs=100, minmax=(-1.3688035702393972, 0.945982920007214), mean=-0.10481901059653059, variance=0.26698104045670723, skewness=-0.2859342660082734, kurtosis=-0.2984214948175521)
    Reader statistics:
     DescribeResult(nobs=2, minmax=(-0.33333333333333337, 0.33333333333333326), mean=-5.551115123125783e-17, variance=0.2222222222222222, skewness=0.0, kurtosis=-2.0)
    Mean reader preference distance:
     0.6666666666666666
    ---------------



    
![plot_2_1](/assets/images/output_2_1.png)
    


    ---------------
    Publication statistics:
     DescribeResult(nobs=2, minmax=(-0.3967149532319961, 0.3563419396806129), mean=-0.020186506775691615, variance=0.28354734198159637, skewness=0.0, kurtosis=-2.0)
    Reader statistics:
     DescribeResult(nobs=100, minmax=(-0.39671495323199635, 0.3563419396806131), mean=-0.08043105820870027, variance=0.13953966162973105, skewness=0.32417635938924266, kurtosis=-1.8949096880131369)
    Mean reader preference distance:
     0.37059527093638683
    Publication reader counts:
     {0: 58, 1: 42}
    ---------------


    
![plot_2_3](/assets/images/output_2_3.png)
    


As is intuitive, we see a convergence between the readers and the publications. One group of readers converges to the negative publication, and a second group converges to the positive one. The publications become slightly more extreme, but there is a centering effect on the readers, especially on those who start off more extreme. Note also that P = 2 reduces the mean preference distance, since groups of readers of the same publication converge to identical preferences.

## TODOs

1. Show a case with large P
2. Understand the numerical impacts of different values for alpha and beta.
3. Consider median reader preference distance
3. Add a "sensationalism" coefficient to the publications, and show how for a large P, this causes a drift towards extrema for readers.
