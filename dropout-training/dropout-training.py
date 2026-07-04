import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x)
    if rng is None:
        rng = np.random.default_rng()
    pr = 1/ (1-p)
    factor = rng.choice([0, pr], size = x.shape, p = [p, 1-p])
    return x * factor, factor