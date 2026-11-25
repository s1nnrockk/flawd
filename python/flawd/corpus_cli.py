"""Command line entry point for exporting FLAWD token streams."""
import argparse
import json

from .corpus import tokenize_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert FLAWD JSONL text into a binary token stream.")
    parser.add_argument("input", help="Clean UTF-8 JSONL dataset")
    parser.add_argument("output", help="Output .tokens file")
    args = parser.parse_args()
    print(json.dumps(tokenize_jsonl(\n        args.input, args.output,\n        source_description=args.source_description,\n        license_id=args.license_id,\n        language=args.language,\n    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
