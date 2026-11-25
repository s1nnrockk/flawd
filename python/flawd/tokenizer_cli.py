"""Train and save a deterministic byte-pair tokenizer from cleaned JSONL."""
import argparse
import json

from .dataset import read_jsonl
from .tokenizer import ByteBPETokenizer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer from a cleaned FLAWD JSONL dataset."
    )
    parser.add_argument("input", help="Clean UTF-8 JSONL dataset")
    parser.add_argument("output", help="Output .bpe tokenizer file")
    parser.add_argument("--vocab-size", type=int, default=1024,
                        help="Target vocabulary size including special tokens (default: 1024)")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="Minimum pair frequency for a merge (default: 2)")
    parser.add_argument("--max-training-bytes", type=int, default=2_000_000,
                        help="Maximum UTF-8 bytes used to train the tokenizer")
    args = parser.parse_args()
    records = read_jsonl(args.input)
    tokenizer = ByteBPETokenizer.train(
        (record["text"] for record in records),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_training_bytes=args.max_training_bytes,
    )
    tokenizer.save(args.output)
    print(json.dumps({
        "tokenizer": tokenizer.NAME,
        "vocabulary_size": tokenizer.VOCAB_SIZE,
        "merge_count": len(tokenizer.merges),
        "output": args.output,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
