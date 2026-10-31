
import argparse
import pickle
import time
import resource
import platform

from pathlib import Path

# Reuse the function that already passes your tests.
from tests.adapters import run_train_bpe


def train(dataset, input_path, vocab_size, special_tokens, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Start timing
    start = time.perf_counter()

    # 2. Train BPE tokenizer
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    elapsed = time.perf_counter() - start

    # 3. Measure peak process memory
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    if platform.system() == "Darwin":
        peak_gb = peak_rss / (1024 ** 3)
    else:
        peak_gb = peak_rss / (1024 ** 2)

    # 4. Save vocabulary and merges
    with open(output_dir / "vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    with open(output_dir / "merges.pkl", "wb") as f:
        pickle.dump(merges, f)

    # 5. Find longest token
    longest = max(vocab.values(), key=len)

    special_bytes = {s.encode("utf-8") for s in special_tokens}
    regular_tokens = [
        token for token in vocab.values()
        if token not in special_bytes
    ]
    longest_regular = max(regular_tokens, key=len)

    # 6. Print results
    print(f"Dataset: {dataset}")
    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Peak process memory: {peak_gb:.2f} GB")
    print(f"Vocab size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")

    print(f"Longest token ({len(longest)} bytes):")
    print(repr(longest))
    print(longest.decode("utf-8", errors="replace"))

    print(f"Longest non-special token ({len(longest_regular)} bytes):")
    print(repr(longest_regular))
    print(longest_regular.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["tinystories", "owt"],
        required=True,
    )
    parser.add_argument("--input", default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    configs = {
        "tinystories": {
            "input_path": "data/TinyStoriesV2-GPT4-train.txt",
            "vocab_size": 10000,
            "special_tokens": ["<|endoftext|>"],
        },
        "owt": {
            "input_path": "data/owt_train.txt",
            "vocab_size": 32000,
            "special_tokens": [],
        },
    }

    config = configs[args.dataset]

    train(
        dataset=args.dataset,
        input_path=args.input or config["input_path"],
        vocab_size=args.vocab_size or config["vocab_size"],
        special_tokens=config["special_tokens"],
        output_dir=args.output_dir or f"artifacts/{args.dataset}",
    )
