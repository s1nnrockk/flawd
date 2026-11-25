"""Dependency-free JSONL dataset validation helpers."""
import json
from pathlib import Path
from typing import Iterator

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
            yield {"text": record["text"]}

def validate_jsonl(path: str | Path) -> int:
    return sum(1 for _ in read_jsonl(path))
