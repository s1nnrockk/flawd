"""Command line interface for FLAWD web research and local retrieval."""
from __future__ import annotations

import argparse
import json

from .websearch import KnowledgeStore, research


def main() -> int:
    parser = argparse.ArgumentParser(description="Search the web and build a local FLAWD knowledge index.")
    parser.add_argument("query", help="Question or search phrase")
    parser.add_argument("--db", default="flawd_knowledge.sqlite3", help="SQLite knowledge database")
    parser.add_argument("--limit", type=int, default=5, help="Search result count (1-10)")
    parser.add_argument("--fetch", type=int, default=3, help="How many result pages to read (1-5)")
    parser.add_argument("--local-only", action="store_true", help="Search the saved local knowledge index only")
    args = parser.parse_args()
    store = KnowledgeStore(args.db)
    try:
        if args.local_only:
            result = {"query": args.query, "knowledge": store.search(args.query, args.limit)}
        else:
            result = research(args.query, store, args.limit, args.fetch)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
