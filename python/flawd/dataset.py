"""JSONL dataset reading, validation, normalization, and deduplication."""
import json
import random
import re
from pathlib import Path
from typing import Any, Iterator

_ROLE_PATTERN = re.compile(r"<\|(system|user|assistant)\|>\n")


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("dataset text must be a string")
    return re.sub(r"\s+", " ", text).strip()


def validate_serialized_conversation(text: str) -> None:
    """Validate serialized role markers and reject malformed/injected turns."""
    matches = list(_ROLE_PATTERN.finditer(text))
    if not matches or matches[0].start() != 0:
        raise ValueError("conversation must start with a role marker")
    roles = [match.group(1) for match in matches]
    if roles[0] == "system":
        if roles.count("system") != 1 or len(roles) < 3:
            raise ValueError("system message is allowed only once at the beginning")
        roles = roles[1:]
    if not roles or roles[0] != "user" or roles[-1] != "assistant":
        raise ValueError("conversation must start with user and end with assistant")
    for index, role in enumerate(roles):
        expected = "user" if index % 2 == 0 else "assistant"
        if role != expected:
            raise ValueError("conversation messages must alternate user and assistant")
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if not text[match.end():end].strip():
            raise ValueError("conversation messages must have non-empty content")


def format_conversation(messages: list[dict[str, str]]) -> str:
    """Serialize role-tagged messages into the format used by FLAWD chat prompts."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    parts: list[str] = []
    roles: list[str] = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise ValueError(f"message {index} has an unsupported role")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"message {index} content must be a non-empty string")
        if any(marker in content for marker in ("<|system|>", "<|user|>", "<|assistant|>")):
            raise ValueError(f"message {index} contains a reserved role marker")
        if role == "system" and (index != 1 or "system" in roles):
            raise ValueError("system message is allowed only once at the beginning")
        content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
        parts.append(f"<|{role}|>\n{content}\n")
        roles.append(role)
    result = "".join(parts)
    validate_serialized_conversation(result)
    return result


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read normalized text while retaining a small allowlist of provenance fields."""
    file_path = Path(path)
    provenance_keys = ("source", "license", "source_split", "languages", "category",
                       "sources", "licenses", "source_splits", "categories")
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
                raw_text = record["text"]
                normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
                if normalized.startswith(("<|system|>\n", "<|user|>\n", "<|assistant|>\n")):
                    try:
                        validate_serialized_conversation(normalized)
                    except ValueError as exc:
                        raise ValueError(f"{file_path}:{line_number}: {exc}") from exc
                    text = normalized
                else:
                    text = normalize_text(raw_text)
            elif isinstance(record.get("messages"), list):
                try:
                    text = format_conversation(record["messages"])
                except ValueError as exc:
                    raise ValueError(f"{file_path}:{line_number}: {exc}") from exc
            else:
                raise ValueError(f"{file_path}:{line_number}: expected a string 'text' or a 'messages' array")
            if text:
                cleaned = {"text": text}
                for key in provenance_keys:
                    if key in record:
                        cleaned[key] = record[key]
                yield cleaned


def merge_provenance(target: dict[str, Any], incoming: dict[str, Any]) -> None:
    """Merge attribution metadata when identical text occurs in multiple sources."""
    for singular, plural in (
        ("source", "sources"),
        ("license", "licenses"),
        ("source_split", "source_splits"),
        ("category", "categories"),
    ):
        values = []
        for record in (target, incoming):
            current = record.get(plural, [])
            if isinstance(current, list):
                values.extend(str(value) for value in current if value)
            value = record.get(singular)
            if value:
                values.append(str(value))
        unique = list(dict.fromkeys(values))
        if unique:
            target[singular] = unique[0]
        if len(unique) > 1:
            target[plural] = unique
    languages = []
    for record in (target, incoming):
        current = record.get("languages", [])
        if isinstance(current, list):
            languages.extend(str(value) for value in current if value)
    if languages:
        target["languages"] = sorted(set(languages))


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
    positions: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    input_count = 0
    for record in read_jsonl(source):
        input_count += 1
        text = record["text"]
        if text not in positions:
            positions[text] = len(records)
            records.append(dict(record))
        else:
            merge_provenance(records[positions[text]], record)

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if validation_destination is None:
        source_path = Path(source).resolve()
        if destination_path.resolve() == source_path:
            raise ValueError("Source and output paths must be different")
        with destination_path.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {
            "input_records": input_count,
            "output_records": len(records),
            "duplicates_removed": input_count - len(records),
        }

    validation_path = Path(validation_destination)
    source_path = Path(source).resolve()
    if destination_path.resolve() == validation_path.resolve():
        raise ValueError("Training and validation output paths must be different")
    if source_path in {destination_path.resolve(), validation_path.resolve()}:
        raise ValueError("Source and output paths must be different")
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
            for index, record in enumerate(records):
                output = validation_output if index in validation_indices else train_output
                output.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "input_records": input_count,
        "output_records": len(records),
        "duplicates_removed": input_count - len(records),
        "training_records": len(records) - validation_count,
        "validation_records": validation_count,
        "validation_ratio": validation_ratio,
        "seed": seed,
    }
