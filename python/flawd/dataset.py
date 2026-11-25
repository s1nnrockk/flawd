"""JSONL dataset reading, validation, normalization, and deduplication."""
import json
import random
import re
from pathlib import Path
from typing import Iterator


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("dataset text must be a string")
    return re.sub(r"\s+", " ", text).strip()


def format_conversation(messages: list[dict[str, str]]) -> str:
    """Serialize role-tagged messages into the format used by FLAWD chat prompts."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    parts: list[str] = []
    has_user = False
    has_assistant = False
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"message {index} has an unsupported role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} content must be a non-empty string")
        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        parts.append(f"<|{role}|>\n{content}\n")
        has_user = has_user or role == "user"
        has_assistant = has_assistant or role == "assistant"
    if not has_user or not has_assistant:
        raise ValueError("conversation must contain at least one user and one assistant message")
    return "".join(parts)


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
            if not isinstance(record, dict):
                raise ValueError(f"{file_path}:{line_number}: expected a JSON object")
            if isinstance(record.get("text"), str):
                text = normalize_text(record["text"])
            elif isinstance(record.get("messages"), list):
                try:
                    text = format_conversation(record["messages"])
                except ValueError as exc:
                    raise ValueError(f"{file_path}:{line_number}: {exc}") from exc
            else:
                raise ValueError(f"{file_path}:{line_number}: expected a string 'text' or a 'messages' array")
            if text:
                yield {"text": text}


def validate_jsonl(path: str | Path) -> int:
    return sum(1 for _ in read_jsonl(path))


def build_clean_dataset(
    source: str | Path,
    destination: str | Path,
    validation_destination: str | Path | None = None,
    validation_ratio: float = 0.05,
    seed: int = 2026,
) -> dict[str, int | float]:
    """Normalize and exact-deduplicate records, optionally making a reproducible split."""
    seen: set[str] = set()
    records: list[str] = []
    input_count = 0
    for record in read_jsonl(source):
        input_count += 1
        text = record["text"]
        if text not in seen:
            seen.add(text)
            records.append(text)

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if validation_destination is None:
        with destination_path.open("w", encoding="utf-8", newline="\n") as output:
            for text in records:
                output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        return {
            "input_records": input_count,
            "output_records": len(records),
            "duplicates_removed": input_count - len(records),
        }

    validation_path = Path(validation_destination)
    if destination_path.resolve() == validation_path.resolve():
        raise ValueError("Training and validation output paths must be different")
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    if len(records) < 2:
        raise ValueError("At least two unique records are required for a train/validation split")

    validation_count = max(1, min(len(records) - 1, round(len(records) * validation_ratio)))
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_indices = set(indices[:validation_count])
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w", encoding="utf-8", newline="\n") as train_output:
        with validation_path.open("w", encoding="utf-8", newline="\n") as validation_output:
            for index, text in enumerate(records):
                output = validation_output if index in validation_indices else train_output
                output.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    return {
        "input_records": input_count,
        "output_records": len(records),
        "duplicates_removed": input_count - len(records),
        "training_records": len(records) - validation_count,
        "validation_records": validation_count,
        "validation_ratio": validation_ratio,
        "seed": seed,
    }
