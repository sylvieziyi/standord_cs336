import torch
import torch.nn as nn

from cs336_basics.transformer import attention
from cs336_basics.transformer import RotaryPositionalEmbedding

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias = False)
        self.k_proj = nn.Linear(d_model, d_model, bias = False)
        self.v_proj = nn.Linear(d_model, d_model, bias = False)
        self.o_proj = nn.Linear(d_model, d_model, bias = False)

        if theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta = theta,
                d_k = self.head_dim,
                max_seq_len = max_seq_len,
            )
        else:
            self.rope = None
    
    def forward(
        self, 
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def split_heads(t):
            return t.reshape(
                *t.shape[:-1],
                self.num_heads,
                self.head_dim,
            ).transpose(-3, -2)
        
        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len, device=x.device
                )
            else:
                # Insert a dimension to broadcast over heads
                token_positions = token_positions.unsqueeze(-2)

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        )

        output = attention(
            q, k, v, mask
        )

        output = output.transpose(-3, -2).contiguous()
        output = output.reshape(
            *x.shape[:-2],
            seq_len,
            self.d_model,
        )

        # 7. Output projection
        return self.o_proj(output)