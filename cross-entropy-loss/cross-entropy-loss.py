import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    logarit = 0
    for x in y_pred:
        logarit += np.log(max(x))
    cross_entropy = -1 / n * (logarit)
    return  cross_entropy