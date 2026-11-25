"""Export cleaned records as the FLAWDTOK binary training corpus."""
import json
import os
import struct
import tempfile
from pathlib import Path

from .dataset import read_jsonl
from .tokenizer import ByteBPETokenizer, ByteTokenizer

_MAGIC = b"FLAWDTOK"
_VERSION = 2
_HEADER = struct.Struct("<8sIIQ")
_TOKEN = struct.Struct("<HB")
_ROLE_MARKERS = ("<|system|>\n", "<|user|>\n", "<|assistant|>\n")


def _encode_record(text: str, tokenizer: ByteTokenizer) -> tuple[list[int], list[int]]:
    """Return token IDs and next-token loss flags (1 = include in loss)."""
    if not any(marker in text for marker in _ROLE_MARKERS):
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        return ids, [0] + [1] * (len(ids) - 1)

    mask = bytearray(len(text.encode("utf-8")))
    cursor = 0
    final_role = None
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
        final_role = marker
        content_start = position + len(marker)
        next_positions = [text.find(candidate, content_start) for candidate in _ROLE_MARKERS]
        next_positions = [value for value in next_positions if value >= 0]
        content_end = min(next_positions) if next_positions else len(text)
        if marker == "<|assistant|>\n":
            start_byte = len(text[:content_start].encode("utf-8"))
            end_byte = len(text[:content_end].encode("utf-8"))
            mask[start_byte:end_byte] = b"\x01" * (end_byte - start_byte)
        cursor = content_end

    encoded_text = text.encode("utf-8")
    body_ids: list[int] = []
    body_flags: list[int] = []
    cursor = 0
    while cursor < len(encoded_text):
        segment_flag = mask[cursor]
        end = cursor + 1
        while end < len(encoded_text) and mask[end] == segment_flag:
            end += 1
        segment_ids = tokenizer.encode_bytes(encoded_text[cursor:end])
        body_ids.extend(segment_ids)
        body_flags.extend([int(segment_flag)] * len(segment_ids))
        cursor = end
    ids = [tokenizer.BOS_ID, *body_ids, tokenizer.EOS_ID]
    flags = [0, *body_flags, int(final_role == "<|assistant|>\n")]
    return ids, flags


def tokenize_jsonl(
    source: str | Path,
    destination: str | Path,
    *,
    source_description: str | None = None,
    license_id: str | None = None,
    language: str | None = None,
    tokenizer_path: str | Path | None = None,
) -> dict[str, int]:
    tokenizer = ByteBPETokenizer.load(tokenizer_path) if tokenizer_path else ByteTokenizer()
    output_path = Path(destination)
    source_path = Path(source).resolve()
    if output_path.resolve() == source_path:
        raise ValueError("Source and token output paths must be different")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    token_count = 0
    supervised_tokens = 0
    temporary_token_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=output_path.parent,
            prefix=output_path.name + ".", suffix=".tmp", delete=False,
        ) as output:
            temporary_token_path = Path(output.name)
            output.write(_HEADER.pack(_MAGIC, _VERSION, tokenizer.VOCAB_SIZE, 0))
            for record in read_jsonl(source):
                ids, flags = _encode_record(record["text"], tokenizer)
                for token_id, flag in zip(ids, flags):
                    output.write(_TOKEN.pack(token_id, flag))
                record_count += 1
                token_count += len(ids)
                supervised_tokens += sum(flags)
            if record_count == 0:
                raise ValueError("Cannot export an empty dataset")
            if supervised_tokens == 0:
                raise ValueError("Dataset contains no supervised training targets")
            output.seek(0)
            output.write(_HEADER.pack(_MAGIC, _VERSION, tokenizer.VOCAB_SIZE, token_count))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_token_path, output_path)
    except Exception:
        if temporary_token_path is not None:
            temporary_token_path.unlink(missing_ok=True)
        raise

    manifest = {
        "format": "FLAWDTOK",
        "version": _VERSION,
        "tokenizer": tokenizer.NAME,
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
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    temporary_manifest_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=manifest_path.parent,
            prefix=manifest_path.name + ".", suffix=".tmp", delete=False,
        ) as manifest_file:
            temporary_manifest_path = Path(manifest_file.name)
            manifest_file.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary_manifest_path, manifest_path)
    except Exception:
        if temporary_manifest_path is not None:
            temporary_manifest_path.unlink(missing_ok=True)
        raise
    return {"records": record_count, "tokens": token_count, "supervised_tokens": supervised_tokens}
