import numpy as np

def tanh(x: list) -> np.ndarray:
    """Return tanh applied elementwise to x."""
    # Write code here
    x = np.asarray(x)
    tanh = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return tanh