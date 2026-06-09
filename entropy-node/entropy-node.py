import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    a, counts = np.unique(y, return_counts = True)
    x = [i / len(y) for i in counts]
    H_S = -np.sum(x * np.log2(x))
    return H_S