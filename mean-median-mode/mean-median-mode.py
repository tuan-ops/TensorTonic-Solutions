import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = np.asarray(x)
    mean = np.mean(x)
    median = np.median(x)
    c = Counter(x)
    max_c = np.max(list(c.values()))
    modes = [key for key, cnt in c.items() if cnt == max_c]
    mode = float(min(modes))
    return float(mean),float(median), mode