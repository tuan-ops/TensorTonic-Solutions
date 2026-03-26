import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    """
    Prepend learnable [CLS] token to patch sequence.
    """
    n, N, D = patches.shape
    cls = np.random.randn(1, 1, D )
    cls_token = np.tile(cls, (n ,1, 1))
    patches = np.concatenate([cls_token, patches], axis = 1)
    return patches
    # YOUR CODE HERE
