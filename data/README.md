# FLAWD data

Training corpora and model checkpoints are not committed to this repository.

The supported dataset format is UTF-8 JSON Lines (.jsonl), with one JSON object per line. Records may contain a plain string "text" field or a "messages" array with system/user/assistant roles. Conversation records must alternate user and assistant turns, may include one system message first, and must end with an assistant response.

Example record:
{"text":"A sample training passage."}

Keep private, licensed, or personally identifying data out of public repositories.
Validate datasets before training and record their source and usage permissions.

Conversation example:
{"messages":[{"role":"system","content":"Be accurate."},{"role":"user","content":"Explain a loop."},{"role":"assistant","content":"A loop repeats instructions."}]}

## Token corpus formats

- FLAWDTOK v1: legacy uint16 token IDs without loss masks.
- FLAWDTOK v2: uint16 token IDs plus uint8 assistant-only loss-mask flags, with the original byte tokenizer.
- FLAWDTOK v3: the v2 token payload plus a 64-bit fingerprint identifying the exact BPE merge table.

Use the same BPE tokenizer file to export training and validation splits. The C++
trainer rejects mismatched tokenizer fingerprints, and model checkpoints retain
the fingerprint so local inference cannot silently load a different merge table.
