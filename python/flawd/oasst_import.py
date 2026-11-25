"""Import the public OpenAssistant OASST1 corpus into FLAWD JSONL.

Install the optional dependency with: pip install -e ".[datasets]"
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .dataset import format_conversation

DATASET_ID = "OpenAssistant/oasst1"
LANGUAGE_MAP = {"en": "en", "ru": "ru"}
ROLE_MAP = {"prompter": "user", "assistant": "assistant"}
_RESERVED = re.compile(r"<\\|(system|user|assistant)\\|>")


def _usable(row: dict[str, Any], languages: set[str]) -> bool:
    lang = str(row.get("lang") or "").lower().split("-")[0]
    if lang not in languages or row.get("deleted") is True:
        return False
    if row.get("review_result") is False:
        return False
    text = row.get("text")
    if not isinstance(text, str) or not text.strip() or _RESERVED.search(text):
        return False
    return row.get("role") in ROLE_MAP


def _rank_key(row: dict[str, Any]) -> tuple[int, str]:
    rank = row.get("rank")
    try:
        rank_value = int(rank) if rank is not None else 10**9
    except (TypeError, ValueError):
        rank_value = 10**9
    return rank_value, str(row.get("message_id") or "")


def _best_path(rows: list[dict[str, Any]], languages: set[str]) -> list[dict[str, str]] | None:
    by_id = {str(row.get("message_id")): row for row in rows if row.get("message_id")}
    children: dict[str | None, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parent = row.get("parent_id")
        children[str(parent) if parent else None].append(row)

    roots = [row for row in children.get(None, []) if row.get("role") == "prompter"]
    if not roots:
        return None
    current = min(roots, key=_rank_key)
    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    while current:
        message_id = str(current.get("message_id") or "")
        if not message_id or message_id in seen or not _usable(current, languages):
            return None
        seen.add(message_id)
        path.append(current)
        candidates = [
            child for child in children.get(message_id, [])
            if str(child.get("message_id") or "") in by_id and _usable(child, languages)
        ]
        if not candidates:
            break
        current = min(candidates, key=_rank_key)

    if len(path) < 2 or path[0].get("role") != "prompter" or path[-1].get("role") != "assistant":
        return None
    messages: list[dict[str, str]] = []
    expected = "prompter"
    for row in path:
        role = row.get("role")
        if role != expected:
            return None
        messages.append({"role": ROLE_MAP[role], "content": str(row["text"]).strip()})
        expected = "assistant" if expected == "prompter" else "prompter"
    try:
        format_conversation(messages)
    except ValueError:
        return None
    return messages


def convert_split(dataset_split: Iterable[dict[str, Any]], languages: set[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    trees: dict[str, list[dict[str, Any]]] = defaultdict(list)
    skipped_rows = 0
    for row in dataset_split:
        tree_id = str(row.get("message_tree_id") or "")
        if not tree_id:
            skipped_rows += 1
            continue
        trees[tree_id].append(dict(row))

    output: list[dict[str, Any]] = []
    skipped_trees = 0
    for tree_rows in trees.values():
        messages = _best_path(tree_rows, languages)
        if messages is None:
            skipped_trees += 1
            continue
        output.append({
            "messages": messages,
            "source": DATASET_ID,
            "source_split": str(tree_rows[0].get("_flawd_split", "unknown")),
            "languages": sorted({
                str(row.get("lang") or "").lower().split("-")[0]
                for row in tree_rows if row.get("lang")
            } & languages),
        })
    return output, {
        "source_rows": sum(len(rows) for rows in trees.values()),
        "conversation_trees": len(trees),
        "exported_conversations": len(output),
        "skipped_trees": skipped_trees,
        "rows_without_tree_id": skipped_rows,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and prepare OASST1 for FLAWD.")
    parser.add_argument("--train-output", default="../data/external/oasst1-train.jsonl")
    parser.add_argument("--validation-output", default="../data/external/oasst1-validation.jsonl")
    parser.add_argument("--languages", default="ru,en", help="Comma-separated language codes (default: ru,en)")
    parser.add_argument("--seed-jsonl", help="Optional curated FLAWD JSONL to append to training only")
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset ID")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('Missing optional dependency. Run: pip install -e ".[datasets]"') from exc

    languages = {item.strip().lower() for item in args.languages.split(",") if item.strip()}
    unsupported = languages - set(LANGUAGE_MAP)
    if unsupported or not languages:
        parser.error(f"Supported languages are: {', '.join(sorted(LANGUAGE_MAP))}")

    dataset = load_dataset(args.dataset)
    if "train" not in dataset or "validation" not in dataset:
        raise SystemExit("Expected official train and validation splits; refusing to create a random split.")

    train_rows, train_stats = convert_split(dataset["train"], languages)
    validation_rows, validation_stats = convert_split(dataset["validation"], languages)

    if args.seed_jsonl:
        from .dataset import read_jsonl
        seed_records = list(read_jsonl(args.seed_jsonl))
        train_rows.extend(seed_records)

    train_path = Path(args.train_output)
    validation_path = Path(args.validation_output)
    if train_path.resolve() == validation_path.resolve():
        raise SystemExit("Training and validation outputs must be different files.")
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)

    report = {
        "dataset": args.dataset,
        "license": "Apache-2.0",
        "languages": sorted(languages),
        "train": train_stats,
        "validation": validation_stats,
        "curated_seed_records_added_to_train": len(seed_records) if args.seed_jsonl else 0,
        "outputs": {"train": str(train_path), "validation": str(validation_path)},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
