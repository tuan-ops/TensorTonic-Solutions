import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """Return matrix normalized along the selected axis."""
    # Write code here
    lst = np.asarray(matrix)
    result = []
    if norm_type == "l1":
        result = np.sum(np.abs(lst), axis = axis)
    elif norm_type == "max":
        result = np.max(np.abs(lst), axis = axis)
    else:
        result = np.sqrt(np.sum(lst ** 2, axis = axis))
    result = np.where(result == 0, 1.0, result)
    if axis == 1:
        return  lst/result[:, None]
    return lst / result