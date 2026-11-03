import torch
import torch.nn as nn
import math

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean = 0.0, std = std, a = -3 * std, b = 3 * std)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @self.W.T