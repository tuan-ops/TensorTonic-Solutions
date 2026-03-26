import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x = np.asarray(x, dtype = float)
    p = np.asarray(p, dtype = float)
    if np.allclose(np.sum(p), 1.0) == False :
        raise ValueError("ValueError")
    
    # Write code here
    exp = np.sum(x * p)
    return exp
