# FLAWD Python tools

Python contains the byte tokenizer and dataset preparation tools. The C++ engine
contains the neural network, Transformer, optimizer, and training loop.

## Requirements

Python 3.10 or newer. The baseline package has no runtime dependencies.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Dataset format

UTF-8 JSON Lines. Plain-text records are supported:

```json
{"text":"A training passage with documented usage rights."}
```

Instruction-tuning conversations can use role-tagged messages. The serializer
preserves message line breaks (important for code examples):

```json
{"messages":[{"role":"system","content":"Be helpful and accurate."},{"role":"user","content":"Explain this function."},{"role":"assistant","content":"It returns a value."}]}
```

Roles currently supported: system, user, and assistant. Conversation data is
serialized using role markers, the same markers used by the chat prompt.

## Prepare a corpus

```powershell
python -m flawd.dataset_cli raw.jsonl clean.jsonl
python -m flawd.corpus_cli clean.jsonl clean.tokens
```

The first command validates records, normalizes whitespace, and removes exact
duplicates. It can also create a deterministic document-level holdout split:

```powershell
python -m flawd.dataset_cli raw.jsonl train.jsonl --validation-output validation.jsonl --validation-ratio 0.05 --seed 2026
python -m flawd.corpus_cli train.jsonl train.tokens
python -m flawd.corpus_cli validation.jsonl validation.tokens
```

The second command writes a little-endian FLAWDTOK binary token stream and a JSON
manifest. Train and validation corpora stay separate to avoid measuring training
examples as held-out quality. The C++ `flawd_train` executable can report the
validation loss when both token files are supplied.

## Tokenizer

ByteTokenizer maps UTF-8 bytes to IDs 0–255 and reserves 256, 257, and 258 for
PAD, BOS, and EOS. EOS is appended to each exported text record. This transparent
baseline can later be replaced by a subword tokenizer.
