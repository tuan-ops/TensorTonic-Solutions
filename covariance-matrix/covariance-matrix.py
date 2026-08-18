import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.shape[0] - 1 == 0 or X.ndim == 1: return None
    mean = np.mean(X, axis = 0, keepdims = True)
    X_center = X - mean
    cov = 1/(X.shape[0] - 1) * np.dot(X_center.T, X_center)
    return cov