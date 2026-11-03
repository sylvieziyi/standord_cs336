import torch
import torch.nn as nn

class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.Embedding = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = 1.0
        nn.init.trunc_normal_(self.Embedding, mean = 0.0, std = std, a = -3 * std, b = 3 * std)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.Embedding[token_ids]