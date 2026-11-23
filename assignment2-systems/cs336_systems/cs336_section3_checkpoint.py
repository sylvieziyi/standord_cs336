"""CS336 A2 §3: run on CPU or Apple Silicon MPS; no CUDA required.

Compare ordinary autograd and non-reentrant activation checkpointing.
The saved-tensor numbers are logical hook observations, NOT peak GPU memory.
"""
import argparse
import copy
import gc
import time

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class TinyBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, 4 * width)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * width, width)
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class FourBlocks(nn.Module):
    def __init__(self, width: int, depth: int = 4, chunk_size: int = 0):
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock(width) for _ in range(depth))
        self.chunk_size = chunk_size

    def forward(self, x):
        if self.chunk_size == 0:
            for block in self.blocks:
                x = block(x)
            return x

        for start in range(0, len(self.blocks), self.chunk_size):
            end = min(start + self.chunk_size, len(self.blocks))

            def segment(inp, first=start, last=end):
                for i in range(first, last):
                    inp = self.blocks[i](inp)
                return inp

            x = checkpoint(segment, x, use_reentrant=False)
        return x


def synchronize(device: str):
    if device == 'mps':
        torch.mps.synchronize()


def mps_allocated_mb(device: str):
    if device == 'mps':
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return None


def run_case(label, model, x, device):
    counts = {'n': 0, 'bytes': 0}
    shapes = []

    def pack(t):
        counts['n'] += 1
        counts['bytes'] += t.numel() * t.element_size()
        if len(shapes) < 8:
            shapes.append(tuple(t.shape))
        return t

    def unpack(t):
        return t

    model.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    before = mps_allocated_mb(device)
    synchronize(device)
    start = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        y = model(x)
        loss = y.square().mean()
        synchronize(device)
        forward_ms = (time.perf_counter() - start) * 1000
        after_forward = mps_allocated_mb(device)
        start = time.perf_counter()
        loss.backward()
        synchronize(device)
        backward_ms = (time.perf_counter() - start) * 1000
    after_backward = mps_allocated_mb(device)

    print(f'\n{label}')
    print(f'  Forward: {forward_ms:.2f} ms | Backward: {backward_ms:.2f} ms')
    print(f'  Saved tensors during forward: {counts["n"]} calls, '
          f'{counts["bytes"] / (1024 ** 2):.2f} MiB logical sizes (may share storage)')
    print(f'  First saved shapes: {shapes}')
    print(f'  Per-block forward calls (includes recomputation): '
          f'{[b.forward_calls for b in model.blocks]}')
    if before is not None:
        print(f'  MPS allocated snapshots: before={before:.1f} MiB, '
              f'after forward={after_forward:.1f} MiB, '
              f'after backward={after_backward:.1f} MiB')
        print('  (These snapshots are NOT peak memory, especially during recomputation.)')
    return y.detach().cpu(), {name: p.grad.detach().cpu().clone()
                              for name, p in model.named_parameters()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default='auto')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--seq', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    args = parser.parse_args()
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available; run with --device cpu')

    torch.manual_seed(123)
    template = FourBlocks(args.width, depth=4).to(device)
    x = torch.randn(args.batch, args.seq, args.width, device=device)
    print(f'Device={device} | x={tuple(x.shape)} | 4 blocks | float32')
    reference_y, reference_grads = None, None
    for name, chunk in [('No checkpoint', 0), ('2-block segments', 2), ('1-block segments', 1)]:
        model = copy.deepcopy(template)
        model.chunk_size = chunk
        y, grads = run_case(name, model, x, device)
        if reference_y is None:
            reference_y, reference_grads = y, grads
        else:
            diff = (y - reference_y).abs().max().item()
            grad_diff = max((grads[k] - reference_grads[k]).abs().max().item()
                            for k in grads)
            print(f'  Numerical check: max output diff={diff:.2e}, '
                  f'max parameter-grad diff={grad_diff:.2e}')
        del model, y, grads
        gc.collect()
    print('\nNote: the first run of each mode can include warm-up costs; repeat for stable timings.')


if __name__ == '__main__':
    main()
