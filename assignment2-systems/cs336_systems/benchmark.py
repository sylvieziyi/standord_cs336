import argparse
import statistics
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

def choose_device(choice: str) -> torch.device:
    if choice != "auto":
        device = torch.device(choice)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this computer")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available on this computer")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def synchronize(device: torch.device) -> None:
    """Wait for the GPU before reading a wall-clock timer."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_step(model, optimizer, x, target, mode, device):
    """Return per-phase wall-clock timings in milliseconds.

    Note: gradient clearing happens before timed regions; phase times include
    GPU synchronizations. This is useful for comparing configurations, not a
    pure-kernel microbenchmark.
    """
    timings = {}
    if mode != "forward":
        optimizer.zero_grad(set_to_none=True)
    synchronize(device)

    t0 = time.perf_counter()
    if mode == "forward":
        with torch.no_grad():
            logits = model(x)
    else:
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    synchronize(device)
    t1 = time.perf_counter()
    timings["forward"] = (t1 - t0) * 1000

    if mode != "forward":
        loss.backward()
        synchronize(device)
        t2 = time.perf_counter()
        timings["backward"] = (t2 - t1) * 1000

        if mode == "full":
            optimizer.step()
            synchronize(device)
            t3 = time.perf_counter()
            timings["optimizer"] = (t3 - t2) * 1000

    timings["total"] = sum(timings.values())
    return timings


def main():
    parser = argparse.ArgumentParser(description="CS336 local benchmarking")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--mode", choices=["all", "forward", "backward", "full"], default="all")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    args = parser.parse_args()
    if args.warmup < 0 or args.steps < 1:
        parser.error("--warmup must be >= 0 and --steps must be >= 1")
    if args.d_model % args.num_heads != 0 or (args.d_model // args.num_heads) % 2:
        parser.error("d_model / num_heads must be an even integer (for RoPE)")

    # Run inside the official assignment2-systems repository. It already
    # contains the reference Transformer implementation from Assignment 1.
    from cs336_basics.model import BasicsTransformerLM

    device = choose_device(args.device)
    torch.manual_seed(42)
    print(f"Device: {device}")
    print(f"Model: layers={args.num_layers}, d_model={args.d_model}, "
          f"heads={args.num_heads}, d_ff={args.d_ff}")
    print(f"Batch={args.batch_size}, context={args.context_length}, vocab={args.vocab_size}")

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    target = torch.randint(0, args.vocab_size, x.shape, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}\n")

    modes = ["forward", "backward", "full"] if args.mode == "all" else [args.mode]
    for mode in modes:
        print(f"[{mode}] warmup={args.warmup}, measured steps={args.steps}")
        for _ in range(args.warmup):
            benchmark_step(model, optimizer, x, target, mode, device)

        samples = defaultdict(list)
        for _ in range(args.steps):
            times = benchmark_step(model, optimizer, x, target, mode, device)
            for key, value in times.items():
                samples[key].append(value)

        for phase in ("forward", "backward", "optimizer", "total"):
            if phase in samples:
                vals = samples[phase]
                avg = statistics.mean(vals)
                sd = statistics.stdev(vals) if len(vals) >= 2 else 0.0
                print(f"  {phase:>9s}: {avg:9.2f} ms  (std {sd:.2f} ms)")
        print()


if __name__ == "__main__":
    main()