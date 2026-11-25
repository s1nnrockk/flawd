"""Command line entry point for exporting FLAWD token streams."""
import argparse
import json

from .corpus import tokenize_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert FLAWD JSONL text into a binary token stream.")
    parser.add_argument("input", help="Clean UTF-8 JSONL dataset")
    parser.add_argument("output", help="Output .tokens file")
    parser.add_argument("--source-description", help="Dataset source or provenance note")
    parser.add_argument("--license", dest="license_id", help="Dataset license or permission identifier")
    parser.add_argument("--language", help="Primary language tag, e.g. en or ru")
    parser.add_argument("--tokenizer", help="Optional trained .bpe tokenizer file")
    args = parser.parse_args()
    print(json.dumps(tokenize_jsonl(
        args.input, args.output,
        source_description=args.source_description,
        license_id=args.license_id,
        language=args.language,
        tokenizer_path=args.tokenizer,
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
