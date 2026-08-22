import numpy as np

def swish(x: list) -> np.ndarray:
    """Return Swish applied elementwise to x."""
    # Write code here
    x = np.asarray(x)
    sigmoid = 1 / (1 + np.exp(-x))
    return x * sigmoid