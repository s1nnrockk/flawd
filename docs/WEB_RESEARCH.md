# FLAWD Web Research

FLAWD separates live retrieval from model training.

## Current implementation

- DuckDuckGo lightweight HTML search, with Bing RSS as a fallback. No API key is required for the initial prototype; public endpoints can rate-limit or change.
- Concurrent retrieval of a small number of result pages.
- HTML extraction strips common navigation/script elements and keeps readable text.
- Local SQLite knowledge store with chunking, URL attribution, content hashes, and FTS5 full-text search when the SQLite build supports it.
- CLI emits structured JSON containing search results, retrieved excerpts, source URLs, and fetch errors.

## Run

From the repository's `python` directory:

```powershell
python -m flawd.websearch_cli "как работает трансформер" --db flawd_knowledge.sqlite3
python -m flawd.websearch_cli "attention mechanism" --db flawd_knowledge.sqlite3 --local-only
```

## Safety and quality boundaries

- Only public HTTP(S) destinations are accepted. Local, private, reserved, and credential-bearing URLs are rejected; redirects are checked too.
- Page size and request time are bounded. Only HTML/XHTML/plain text is extracted.
- Search snippets and pages are untrusted data. They must never be executed as code or followed as instructions.
- A result is not automatically true. A future answer composer should cite the page URL and distinguish source claims from verified facts.
- Search does not update model weights. Saved pages form a local retrieval index (RAG). Training examples should be curated, licensed, deduplicated, quality-filtered, and split by source/document before any training run.
- Do not crawl entire websites indiscriminately. Respect robots.txt, site terms, rate limits, copyright, privacy, and attribution before adding a dedicated crawler.
- The current prototype is a research backend, not yet connected to the chat UI or the model's prompt builder.
