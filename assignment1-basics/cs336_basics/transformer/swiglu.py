import math

import torch
import torch.nn as nn

from cs336_basics.transformer import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil((8 * d_model / 3) / 64) * 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        silu = gate * torch.sigmoid(gate)
        hidden = silu * self.w3(x)

        return self.w2(hidden)
