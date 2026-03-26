import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int) -> np.ndarray:
    """
    Add learnable position embeddings to patch embeddings.
    """
    # YOUR CODE HERE
    n, N, D = patches.shape
    E = np.random.randn(N, D)
    patches = patches + E
    
    return patches