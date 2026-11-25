"""Command line entry point for cleaning FLAWD JSONL datasets."""
import argparse
import json

from .dataset import build_clean_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate, normalize, and deduplicate a FLAWD JSONL dataset.")
    parser.add_argument("input", help="Source UTF-8 JSONL file")
    parser.add_argument("output", help="Destination UTF-8 JSONL file")
    parser.add_argument("--validation-output", help="Optional separate validation JSONL file")
    parser.add_argument("--validation-ratio", type=float, default=0.05, help="Validation fraction (default: 0.05)")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic split seed")
    args = parser.parse_args()
    stats = build_clean_dataset(
        args.input, args.output, args.validation_output, args.validation_ratio, args.seed
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
