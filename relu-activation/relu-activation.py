import numpy as np

def relu(x) -> np.ndarray:
    """Return ReLU applied elementwise to x."""
    # Write code here
    x = np.asarray(x)
    return np.where(x <= 0, 0, x)