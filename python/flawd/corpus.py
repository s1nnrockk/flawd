"""Export cleaned text records as the compact FLAWD token-stream format."""
import json
import struct
from pathlib import Path

from .dataset import read_jsonl
from .tokenizer import ByteTokenizer

_MAGIC = b"FLAWDTOK"
_VERSION = 1
_HEADER = struct.Struct("<8sIIQ")
_TOKEN = struct.Struct("<H")


def tokenize_jsonl(source: str | Path, destination: str | Path) -> dict[str, int]:
    tokenizer = ByteTokenizer()
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_count = 0
    token_count = 0
    with output_path.open("w+b") as output:
        output.write(_HEADER.pack(_MAGIC, _VERSION, tokenizer.VOCAB_SIZE, 0))
        for record in read_jsonl(source):
            ids = tokenizer.encode(record["text"], add_eos=True)
            for token_id in ids:
                output.write(_TOKEN.pack(token_id))
            record_count += 1
            token_count += len(ids)
        output.seek(0)
        output.write(_HEADER.pack(_MAGIC, _VERSION, tokenizer.VOCAB_SIZE, token_count))

    manifest = {
        "format": "FLAWDTOK",
        "version": _VERSION,
        "tokenizer": "utf8-byte-v1",
        "vocabulary_size": tokenizer.VOCAB_SIZE,
        "records": record_count,
        "tokens": token_count,
        "source_name": Path(source).name,
    }
    output_path.with_suffix(output_path.suffix + ".json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"records": record_count, "tokens": token_count}
