import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float, d_k: int, max_seq_len:int, device=None):
        super().__init__()
        self.d_k = d_k
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k)) 

        positions = torch.arange(max_seq_len, device=device)
        angles = positions[:, None] * freqs[None, :]

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)

        x1 = x_pairs[..., 0]
        x2 = x_pairs[..., 1]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos

        rotated = torch.stack([rotated_x1, rotated_x2], dim=-1)

        return rotated.reshape(x.shape)