from __future__ import annotations

import os
from collections import Counter, defaultdict

import regex as re


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    PAT = (
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+"""
        r"""| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    # 1. Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}

    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError("vocab_size is too small")

    # 2. Read corpus
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Treat special tokens as boundaries
    if special_tokens:
        special_pattern = "|".join(
            re.escape(t)
            for t in sorted(special_tokens, key=len, reverse=True)
        )
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]

    # 3. Pre-tokenization and frequency counting
    word_freq = Counter()

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            word = tuple(
                vocab[b] for b in match.group().encode("utf-8")
            )
            word_freq[word] += 1

    # Give every unique pre-token an integer ID
    words = list(word_freq.keys())
    counts = [word_freq[word] for word in words]

    # 4. Initialize pair frequencies and inverted index
    pair_freq = Counter()
    pair_to_words = defaultdict(set)

    for idx, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_freq[pair] += counts[idx]
            pair_to_words[pair].add(idx)

    merges = []

    # 5. BPE training
    while len(vocab) < vocab_size and pair_freq:

        best_pair = max(
            pair_freq,
            key=lambda p: (pair_freq[p], p)
        )

        new_token = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = new_token
        merges.append(best_pair)

        # Only update words containing the best pair
        affected_words = list(pair_to_words[best_pair])

        for idx in affected_words:
            old_word = words[idx]
            freq = counts[idx]

            # Remove contributions of old adjacent pairs
            for pair in zip(old_word, old_word[1:]):
                pair_freq[pair] -= freq
                pair_to_words[pair].discard(idx)

                if pair_freq[pair] == 0:
                    del pair_freq[pair]

            # Apply the merge
            new_word = []
            i = 0

            while i < len(old_word):
                if (
                    i + 1 < len(old_word)
                    and (old_word[i], old_word[i + 1]) == best_pair
                ):
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1

            new_word = tuple(new_word)

            # Add contributions of new adjacent pairs
            for pair in zip(new_word, new_word[1:]):
                pair_freq[pair] += freq
                pair_to_words[pair].add(idx)

            words[idx] = new_word

    return vocab, merges
