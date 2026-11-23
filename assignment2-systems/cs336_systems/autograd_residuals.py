import torch
from torch import nn

device = "mps" if torch.backends.mps.is_available() else "cpu"

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )
        x = x * rms
        return self.weight * x


# Small size for MacBook Air
x = torch.randn(
    2, 128, 128,
    device=device,
    requires_grad=True,
)

model = torch.compile(RMSNorm(128).to(device))


def pack_hook(t):
    size = t.numel() * t.element_size()
    print(f"SAVE  shape={tuple(t.shape)}, bytes={size}")
    return t


def unpack_hook(t):
    print(f"LOAD  shape={tuple(t.shape)}")
    return t


with torch.autograd.graph.saved_tensors_hooks(
    pack_hook, unpack_hook
):
    y = model(x)
    loss = y.sum()
    loss.backward()

print("Finished backward!")