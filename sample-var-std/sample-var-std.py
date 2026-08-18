import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    x = np.asarray(x)
    var = 1/(x.size - 1) * (np.sum((x - np.mean(x))**2))
    return var, np.sqrt(var)