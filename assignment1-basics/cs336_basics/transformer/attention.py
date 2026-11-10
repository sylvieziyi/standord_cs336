import torch
import math
from cs336_basics.transformer import softmax

def attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)
    
    scores = scores / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(
            ~mask.to(device=scores.device, dtype=torch.bool),
            float('-inf'),
        )

    attention_weights = softmax(scores, dim = -1)
    output = attention_weights @ V
    return output
        
