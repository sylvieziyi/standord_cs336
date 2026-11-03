import regex as re


# Problem (unicode2): Unicode Encodings

# (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
# UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
# input strings.

# Reason 1: Better storage efficiency for English and code
# Reason 2: Avoid unnecessary zero bytes during BPE training
# Reason 3: Simpler handling of Unicode and portability

# (b) correct function:
def decode_utf8_bytes_to_str(bytestring: bytes):
    return bytestring.decode("utf-8")


# (c) Give a two-byte sequence that does not decode to any Unicode character(s).
# 0xC3 starts a two-byte UTF-8 sequence, so the next byte must be a continuation byte in the range 0x80–0xBF, but 0x28 is outside that range.
bytestring = b'\xc3\x28'

# Problem (train_bpe): BPE Tokenizer Training




