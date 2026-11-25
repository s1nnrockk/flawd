"""Lossless byte and byte-pair tokenizers for FLAWD."""
from collections import Counter
import os
from pathlib import Path
import struct
import tempfile

_MAGIC = b"FLAWDBPE"
_VERSION = 1
_HEADER = struct.Struct("<8sII")
_MERGE = struct.Struct("<HH")


class ByteTokenizer:
    PAD_ID = 256
    BOS_ID = 257
    EOS_ID = 258
    VOCAB_SIZE = 259
    NAME = "utf8-byte-bos-eos-v2"

    def encode_bytes(self, data: bytes, *, add_bos: bool = False,
                     add_eos: bool = False) -> list[int]:
        ids = list(data)
        if add_bos:
            ids.insert(0, self.BOS_ID)
        if add_eos:
            ids.append(self.EOS_ID)
        return ids

    def encode(self, text: str, *, add_bos: bool = False,
               add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self.encode_bytes(text.encode("utf-8"), add_bos=add_bos, add_eos=add_eos)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        data = bytearray()
        for token_id in token_ids:
            if not isinstance(token_id, int) or not 0 <= token_id < self.VOCAB_SIZE:
                raise ValueError(f"invalid token id: {token_id!r}")
            if token_id <= 255:
                data.append(token_id)
            elif not skip_special_tokens:
                raise ValueError("special tokens cannot be decoded as UTF-8 bytes")
        return data.decode("utf-8", errors="replace")


class ByteBPETokenizer(ByteTokenizer):
    """Deterministic byte-level BPE with a portable binary merge table."""

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = list(merges or [])
        if len(self.merges) > 65536 - self.BOS_ID - 1:
            raise ValueError("BPE vocabulary exceeds the uint16 token format")
        self.VOCAB_SIZE = 259 + len(self.merges)
        self.NAME = "utf8-byte-bpe-v1"
        self._ranks: dict[tuple[int, int], int] = {}
        self._children: list[tuple[int, int] | None] = [None] * self.VOCAB_SIZE
        for index, pair in enumerate(self.merges):
            token_id = 259 + index
            left, right = pair
            if (not isinstance(left, int) or not isinstance(right, int)
                    or left < 0 or right < 0 or left >= token_id or right >= token_id
                    or left in (self.PAD_ID, self.BOS_ID, self.EOS_ID)
                    or right in (self.PAD_ID, self.BOS_ID, self.EOS_ID)):
                raise ValueError(f"invalid BPE merge at index {index}")
            if pair in self._ranks:
                raise ValueError(f"duplicate BPE merge at index {index}")
            self._ranks[pair] = index
            self._children[token_id] = pair

    @classmethod
    def train(cls, texts, *, vocab_size: int = 1024,
              min_frequency: int = 2, max_training_bytes: int = 2_000_000):
        if not isinstance(vocab_size, int) or not 259 <= vocab_size <= 65536:
            raise ValueError("vocab_size must be between 259 and 65536")
        if not isinstance(min_frequency, int) or min_frequency < 2:
            raise ValueError("min_frequency must be at least 2")
        sequences = []
        total_bytes = 0
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("training texts must be strings")
            encoded = text.encode("utf-8")
            total_bytes += len(encoded)
            if total_bytes > max_training_bytes:
                raise ValueError("BPE training sample exceeds max_training_bytes")
            if encoded:
                sequences.append(list(encoded))
        merges = []
        while 259 + len(merges) < vocab_size:
            counts = Counter()
            for sequence in sequences:
                counts.update(zip(sequence, sequence[1:]))
            if not counts:
                break
            pair, frequency = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if frequency < min_frequency:
                break
            new_id = 259 + len(merges)
            merges.append(pair)
            for sequence_index, sequence in enumerate(sequences):
                if len(sequence) < 2:
                    continue
                merged = []
                index = 0
                while index < len(sequence):
                    if (index + 1 < len(sequence)
                            and sequence[index] == pair[0]
                            and sequence[index + 1] == pair[1]):
                        merged.append(new_id)
                        index += 2
                    else:
                        merged.append(sequence[index])
                        index += 1
                sequences[sequence_index] = merged
        return cls(merges)

    def encode_bytes(self, data: bytes, *, add_bos: bool = False,
                     add_eos: bool = False) -> list[int]:
        tokens = list(data)
        while len(tokens) > 1:
            best_rank = None
            best_pair = None
            for left, right in zip(tokens, tokens[1:]):
                rank = self._ranks.get((left, right))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (left, right)
            if best_pair is None:
                break
            merged = []
            index = 0
            while index < len(tokens):
                if (index + 1 < len(tokens)
                        and (tokens[index], tokens[index + 1]) == best_pair):
                    merged.append(259 + best_rank)
                    index += 2
                else:
                    merged.append(tokens[index])
                    index += 1
            tokens = merged
        if add_bos:
            tokens.insert(0, self.BOS_ID)
        if add_eos:
            tokens.append(self.EOS_ID)
        return tokens

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        data = bytearray()
        for token_id in token_ids:
            if not isinstance(token_id, int) or not 0 <= token_id < self.VOCAB_SIZE:
                raise ValueError(f"invalid token id: {token_id!r}")
            if token_id <= 255:
                data.append(token_id)
            elif token_id in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                if not skip_special_tokens:
                    raise ValueError("special tokens cannot be decoded as UTF-8 bytes")
            else:
                pending = [token_id]
                while pending:
                    current = pending.pop()
                    if current <= 255:
                        data.append(current)
                    else:
                        pair = self._children[current]
                        if pair is None:
                            raise ValueError(f"invalid merged token id: {current}")
                        pending.append(pair[1])
                        pending.append(pair[0])
        return data.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b", dir=destination.parent,
                prefix=destination.name + ".", suffix=".tmp", delete=False,
            ) as output:
                temporary = Path(output.name)
                output.write(_HEADER.pack(_MAGIC, _VERSION, self.VOCAB_SIZE))
                for left, right in self.merges:
                    output.write(_MERGE.pack(left, right))
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path):
        source = Path(path)
        with source.open("rb") as input_file:
            header = input_file.read(_HEADER.size)
            if len(header) != _HEADER.size:
                raise ValueError("BPE tokenizer file is truncated")
            magic, version, vocab_size = _HEADER.unpack(header)
            if magic != _MAGIC or version != _VERSION:
                raise ValueError("Unsupported BPE tokenizer file")
            if not 259 <= vocab_size <= 65536:
                raise ValueError("BPE tokenizer vocabulary size is invalid")
            merge_count = vocab_size - 259
            payload = input_file.read()
            if len(payload) != merge_count * _MERGE.size:
                raise ValueError("BPE tokenizer file size does not match its header")
            merges = [_MERGE.unpack_from(payload, index * _MERGE.size)
                      for index in range(merge_count)]
        return cls(merges)
