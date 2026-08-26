import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scale = math.sqrt(K.shape[-1])
    scores = scores / scale 
    atten = torch.matmul((torch.exp(scores))/ (torch.sum(torch.exp(scores), dim = -1, keepdims = True)) , V)
    return torch.Tensor(atten)
        