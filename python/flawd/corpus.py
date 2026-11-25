"""Export cleaned records as the FLAWDTOK binary training corpus."""
import json
import struct
from pathlib import Path

from .dataset import read_jsonl
from .tokenizer import ByteTokenizer

_MAGIC = b"FLAWDTOK"
_VERSION = 2
_HEADER = struct.Struct("<8sIIQ")
_TOKEN = struct.Struct("<HB")
_ROLE_MARKERS = ("<|system|>\n", "<|user|>\n", "<|assistant|>\n")


def _encode_record(text: str, tokenizer: ByteTokenizer) -> tuple[list[int], list[int]]:
    """Return token IDs and next-token loss flags (1 = include in loss)."""
    if not any(marker in text for marker in _ROLE_MARKERS):
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        return ids, [0] + [1] * (len(ids) - 2) + [0]

    mask = bytearray(len(text.encode("utf-8")))
    cursor = 0
    while cursor < len(text):
        next_marker = min(
            ((text.find(marker, cursor), marker) for marker in _ROLE_MARKERS
             if text.find(marker, cursor) >= 0),
            default=(-1, ""),
            key=lambda item: item[0],
        )
        if next_marker[0] < 0:
            break
        position, marker = next_marker
        content_start = position + len(marker)
        next_positions = [text.find(candidate, content_start) for candidate in _ROLE_MARKERS]
        next_positions = [value for value in next_positions if value >= 0]
        content_end = min(next_positions) if next_positions else len(text)
        if marker == "<|assistant|>\n":
            start_byte = len(text[:content_start].encode("utf-8"))
            end_byte = len(text[:content_end].encode("utf-8"))
            mask[start_byte:end_byte] = b"\x01" * (end_byte - start_byte)
        cursor = content_end

    body = tokenizer.encode(text)
    ids = [tokenizer.BOS_ID, *body, tokenizer.EOS_ID]
    flags = [0, *mask, 0]
    return ids, flags


def tokenize_jsonl(
    source: str | Path,
    destination: str | Path,
    *,
    source_description: str | None = None,
    license_id: str | None = None,
    language: str | None = None,
) -> dict[str, int]:
    tokenizer = ByteTokenizer()
    output_path = Path(destination)
    source_path = Path(source).resolve()
    if output_path.resolve() == source_path:
        raise ValueError("Source and token output paths must be different")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    token_count = 0
    supervised_tokens = 0
    with output_path.open("w+b") as output:
        output.write(_HEADER.pack(_MAGIC, _VERSION, tokenizer.VOCAB_SIZE, 0))
        for record in read_jsonl(source):
            ids, flags = _encode_record(record["text"], tokenizer)
            for token_id, flag in zip(ids, flags):
                output.write(_TOKEN.pack(token_id, flag))
            record_count += 1
            token_count += len(ids)
            supervised_tokens += sum(flags)
        output.seek(0)
        output.write(_HEADER.pack(_MAGIC, _VERSION, tokenizer.VOCAB_SIZE, token_count))

    manifest = {
        "format": "FLAWDTOK",
        "version": _VERSION,
        "tokenizer": "utf8-byte-bos-eos-v2",
        "loss_mask": "assistant-content-only-for-role-tagged-records; all-text-for-plain-records",
        "vocabulary_size": tokenizer.VOCAB_SIZE,
        "records": record_count,
        "tokens": token_count,
        "supervised_tokens": supervised_tokens,
        "source_name": Path(source).name,
        "source_description": source_description,
        "license": license_id,
        "language": language,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"records": record_count, "tokens": token_count, "supervised_tokens": supervised_tokens}
