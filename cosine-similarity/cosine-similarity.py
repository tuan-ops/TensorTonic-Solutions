import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.asarray(a)
    b = np.asarray(b)
    if ( (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))) ) == 0:
        return 0
    result = np.dot(a, b) /( (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))) )
    return float(result)