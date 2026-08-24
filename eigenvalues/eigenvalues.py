import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """Return the real eigenvalues in ascending order."""
    # Write code here
    matrix = np.asarray(matrix)
    eigenvalue = np.linalg.eigvals(matrix)
    eigenvalue.sort()
    return np.asarray((eigenvalue.real))