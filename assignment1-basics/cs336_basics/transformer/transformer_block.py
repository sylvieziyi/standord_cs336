import torch
import torch.nn as nn

from cs336_basics.transformer import RMSNorm, SwiGLU, MultiHeadSelfAttention

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ):
        super().__init__()

        # First sub-layer: Attention
        self.ln1 = RMSNorm(d_model, eps=1e-5)

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
        )

        # Second sub-layer: Feed Forward
        self.ln2 = RMSNorm(d_model, eps=1e-5)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # Pre-norm attention + residual
        x = x + self.attn(
            self.ln1(x),
            token_positions,
        )

        # Pre-norm FFN + residual
        x = x + self.ffn(self.ln2(x))

        return x
