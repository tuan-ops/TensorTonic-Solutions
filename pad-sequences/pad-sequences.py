import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    L = max(len(seq) for seq in seqs)
    if max_len is None : max_len = L
    for i, seq in enumerate(seqs):
        if max_len > len(seq):
            seqs[i] = seq + [pad_value] * (max_len - len(seq))
        else: seqs[i] = seq[:max_len]
    return seqs