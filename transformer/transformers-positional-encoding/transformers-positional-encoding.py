import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pos = np.arange(seq_length).reshape(-1, 1)
    pe = np.zeros((seq_length, d_model))
    pe[:,::2] =  np.sin(pos/ (10000 ** (np.arange(0, d_model, 2) /d_model)))
    pe[:,1::2] = np.cos(pos / (10000 ** (np.arange(0, d_model, 2) / d_model)))
    return pe