import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    # Write code here
    pre = np.asarray(predictions)
    result = [np.bincount(pre[:, i]).argmax() for i in range(len(pre[0]))]
    return result