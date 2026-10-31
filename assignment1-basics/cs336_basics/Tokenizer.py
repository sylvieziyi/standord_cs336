from collections.abc import Iterable, Iterator
from pathlib import Path
import pickle
import regex as re

PAT = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = dict(vocab)  # Don't mutate the caller's vocabulary.
        self.merges = list(merges)
        self.bytes_to_id = {value: index for index, value in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}

        self.special_to_id: dict[str, int] = {}
        for token in special_tokens or []:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id
            self.special_to_id[token] = self.bytes_to_id[token_bytes]

        # Longest first: an overlapping special token should win.
        if self.special_to_id:
            alternatives = "|".join(
                re.escape(token)
                for token in sorted(self.special_to_id, key=len, reverse=True)
            )
            self.special_regex = re.compile(alternatives)
        else:
            self.special_regex = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        # These are the files produced by pickle.dump in the training step.
        # Only load pickle files you trust.
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def _segments(self, text: str) -> Iterator[tuple[bool, str, int, int]]:
        """Yield pre-tokens and special tokens, with their character offsets."""
        pos = 0
        if self.special_regex is not None:
            for special_match in self.special_regex.finditer(text):
                # Pre-tokenize the ordinary text before the special token.
                for match in PAT.finditer(text, pos, special_match.start()):
                    yield False, match.group(), match.start(), match.end()
                yield True, special_match.group(), special_match.start(), special_match.end()
                pos = special_match.end()
        for match in PAT.finditer(text, pos):
            yield False, match.group(), match.start(), match.end()

    def _encode_pretoken(self, pretoken: str) -> Iterator[int]:
        # Start with one bytes object per UTF-8 byte.
        tokens = [bytes([value]) for value in pretoken.encode("utf-8")]

        while len(tokens) >= 2:
            # Among all currently adjacent pairs, choose the lowest merge rank.
            pairs = [
                (tokens[i], tokens[i+1]) 
                for i in range(len(tokens) - 1)
            ]

            valid_pairs = []
            for pair in pairs:
                if pair in self.merge_ranks:
                    valid_pairs.append(pair)
            
            if not valid_pairs:
                break
            
            best_pair = min(valid_pairs, key = lambda p: self.merge_ranks[p])

            merged = best_pair[0] + best_pair[1]

            updated = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    updated.append(merged)
                    i += 2
                else:
                    updated.append(tokens[i])
                    i += 1
            tokens = updated

        for token in tokens:
            yield self.bytes_to_id[token]

    def encode(self, text: str) -> list[int]:
        result = []
        for is_special, content, _, _ in self._segments(text):
            if is_special:
                result.append(self.special_to_id[content])
            else:
                result.extend(self._encode_pretoken(content))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Keep an unprocessed tail so pre-tokens/special tokens spanning
        # two arbitrary input chunks are not prematurely split.
        longest_special = max(map(len, self.special_to_id), default=0)
        lookahead = max(64, longest_special + 8)
        buffer = ""

        for chunk in iterable:
            buffer += chunk
            safe_end = len(buffer) - lookahead
            if safe_end <= 0:
                continue

            consumed = 0
            for is_special, content, _, end in self._segments(buffer):
                if end > safe_end:
                    break
                if is_special:
                    yield self.special_to_id[content]
                else:
                    yield from self._encode_pretoken(content)
                consumed = end

            if consumed:
                buffer = buffer[consumed:]

        # The last buffered portion has no further incoming text.
        if buffer:
            yield from self.encode(buffer)

    def decode(self, ids: list[int]) -> str:
        data = b"".join(self.vocab[index] for index in ids)
        return data.decode("utf-8", errors="replace")
