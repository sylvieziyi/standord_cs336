
import torch
import torch.nn as nn

from cs336_basics.transformer import RMSNorm, Embedding, TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
            )
            for _ in range(num_layers)
        ])

        self.ln_final = RMSNorm(d_model, eps=1e-5)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.token_embeddings(token_ids)

        seq_len = token_ids.shape[-1]
        positions = torch.arange(
            seq_len,
            device=token_ids.device,
        )

        # Pass through all Transformer blocks
        for layer in self.layers:
            x = layer(x, positions)

        # Final normalization
        x = self.ln_final(x)

        # (batch, seq_len, d_model)
        # -> (batch, seq_len, vocab_size)
        logits = self.lm_head(x)

        return logits
