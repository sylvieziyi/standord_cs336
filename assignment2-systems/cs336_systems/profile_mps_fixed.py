"""CS336 A2: local profiling on Apple Silicon (VS Code + PyTorch MPS).

From the assignment2-systems project root:
    uv run python cs336_systems/profile_mps.py

Optional true Metal GPU capture (requires full Xcode + compatible PyTorch):
    MTL_CAPTURE_ENABLED=1 uv run python cs336_systems/profile_mps.py --mode forward --metal-trace
    # The script prints the actual generated path; open that exact .gputrace bundle.

The default PyTorch profiler table reports CPU-side operator time, not the
GPU kernel execution times requested by the CUDA/Nsight assignment.
"""

import argparse
import glob
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function


def main() -> None:
    p = argparse.ArgumentParser(description="CS336 local MPS profiling")
    p.add_argument("--mode", choices=("forward", "full"), default="full")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=2048)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--metal-trace", action="store_true",
                   help="Record actual Metal GPU operations to a .gputrace file")
    args = p.parse_args()

    if args.warmup < 0:
        p.error("--warmup must be >= 0")
    if args.d_model % args.num_heads or (args.d_model // args.num_heads) % 2:
        p.error("head dimension must be an even integer (for RoPE)")
    if not torch.backends.mps.is_available():
        p.error("MPS is unavailable. This script is for Apple Silicon Macs")

    from cs336_basics.model import BasicsTransformerLM

    torch.manual_seed(42)
    device = torch.device("mps")
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=10000.0,
    ).to(device)
    model.train(args.mode == "full")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, args.vocab_size,
                      (args.batch_size, args.context_length), device=device)
    target = torch.randint(0, args.vocab_size, x.shape, device=device)

    def step() -> dict[str, float]:
        times = {}
        if args.mode == "full":
            optimizer.zero_grad(set_to_none=True)
        torch.mps.synchronize()

        start = time.perf_counter()
        with record_function("CS336 forward"):
            if args.mode == "forward":
                with torch.no_grad():
                    logits = model(x)
            else:
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), target.reshape(-1)
                )
        torch.mps.synchronize()
        end = time.perf_counter()
        times["forward"] = 1e3 * (end - start)

        if args.mode == "full":
            with record_function("CS336 backward"):
                loss.backward()
            torch.mps.synchronize()
            backward_end = time.perf_counter()
            times["backward"] = 1e3 * (backward_end - end)

            with record_function("CS336 optimizer"):
                optimizer.step()
            torch.mps.synchronize()
            opt_end = time.perf_counter()
            times["optimizer"] = 1e3 * (opt_end - backward_end)
        return times

    print(f"PyTorch: {torch.__version__} | Device: mps | Mode: {args.mode}")
    print(f"Model: {args.num_layers} layers, d_model={args.d_model}, "
          f"sequence length={args.context_length}")
    print(f"Warm-up steps: {args.warmup}")
    for _ in range(args.warmup):
        step()

    if args.metal_trace:
        metal_capture = getattr(torch.mps.profiler, "metal_capture", None)
        if metal_capture is None:
            p.error("This PyTorch version does not expose torch.mps.profiler.metal_capture")
        if not torch.mps.profiler.is_metal_capture_enabled():
            p.error("Metal capture disabled: run with MTL_CAPTURE_ENABLED=1 and full Xcode installed")
        # PyTorch adds its own 0000- prefix and .gputrace suffix. Pass a
        # unique *base name* to avoid collision with older captures.
        trace_name = f"cs336_{args.mode}_{time.time_ns()}"
        print(f"Recording Metal GPU trace with base name: {trace_name}")
        with metal_capture(trace_name):
            times = step()
        matches = glob.glob(f"*-{trace_name}.gputrace")
        print("Open the trace in Xcode's GPU debugger to inspect actual GPU commands.")
        if matches:
            for path in matches:
                print(f"Run: open '{path}'")
        else:
            print(f"No trace found in the current directory; look for *-{trace_name}.gputrace")
    else:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            times = step()
        print("\nTop PyTorch operators by self CPU time (NOT GPU kernel time):")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

    print("\nElapsed phase time (includes MPS synchronization and profiler overhead):")
    for name, value in times.items():
        print(f"  {name:9s}: {value:8.2f} ms")
    print(f"  {'total':9s}: {sum(times.values()):8.2f} ms")
    print("Use your original benchmark.py, not this instrumented run, for baseline timing.")


if __name__ == "__main__":
    main()
