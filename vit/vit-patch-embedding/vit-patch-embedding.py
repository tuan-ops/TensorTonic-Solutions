import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    """
    Convert image to patch embeddings.
    """
    # YOUR CODE HERE
    n, H, W, C = image.shape
    P = patch_size
    D = embed_dim
    patch_flat_size = P * P * C
    num_patches = (H // P) * (W // P)
    x_p = image.reshape(n, H // P, P, W // P, P, C)
    x_p = x_p.transpose(0,1,3,2,4,5).reshape(n, num_patches, -1)
    W = np.random.randn(patch_flat_size, D)
    b = np.random.randn(D)
    z_p = x_p @ W + b
    return z_p
    