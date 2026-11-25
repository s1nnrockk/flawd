"""JSONL dataset reading, validation, normalization, and deduplication."""
import json
import re
from pathlib import Path
from typing import Iterator


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("dataset text must be a string")
    return re.sub(r"\s+", " ", text).strip()


def read_jsonl(path: str | Path) -> Iterator[dict[str, str]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"{file_path}:{line_number}: expected an object with a string 'text' field")
            text = normalize_text(record["text"])
            if text:
                yield {"text": text}


def validate_jsonl(path: str | Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def build_clean_dataset(source: str | Path, destination: str | Path) -> dict[str, int]:
    """Normalize and exact-deduplicate text records into a new JSONL file."""
    seen: set[str] = set()
    input_count = 0
    output_count = 0
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in read_jsonl(source):
            input_count += 1
            text = record["text"]
            if text in seen:
                continue
            seen.add(text)
            output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            output_count += 1
    return {"input_records": input_count, "output_records": output_count,
            "duplicates_removed": input_count - output_count}
