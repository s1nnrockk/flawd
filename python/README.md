# FLAWD Python tools

Python contains the byte tokenizer and dataset preparation tools. The C++ engine
contains the neural network, Transformer, optimizer, and training loop. The CLI trainer averages gradients over four supervised targets per Adam update; incomplete final groups are scaled by their actual size.

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

Conversation-format records use role-tagged messages. A conversation may start with one system message, then alternates user and assistant turns, and must end with an assistant response. Reserved role markers inside message content are rejected. Example:

```json
{"messages":[{"role":"system","content":"Be helpful and accurate."},{"role":"user","content":"Explain this function."},{"role":"assistant","content":"It returns a value."}]}
```

The exporter creates an assistant-only loss mask for role-tagged conversations:
system and user tokens remain in the context, but only assistant content is
counted as a prediction target. Plain-text records retain ordinary next-token
training over their text. This is the first supervised instruction-tuning
primitive; it is not a claim that the tiny baseline has Claude-level capability.

## Prepare a corpus

```powershell
python -m flawd.dataset_cli raw.jsonl clean.jsonl
python -m flawd.corpus_cli clean.jsonl clean.tokens
```

For a deterministic holdout split:

```powershell
python -m flawd.dataset_cli raw.jsonl train.jsonl --validation-output validation.jsonl --validation-ratio 0.05 --seed 2026
python -m flawd.corpus_cli train.jsonl train.tokens
python -m flawd.corpus_cli validation.jsonl validation.tokens
```

The binary FLAWDTOK v2 stream uses a 24-byte little-endian header followed by
one uint16 token ID and one uint8 loss-mask flag per token. Version 1 files
remain readable by the C++ trainer and are treated as unmasked text corpora.
The JSON manifest records source provenance, token count, and supervised-token
count. Train and validation corpora must remain separate.

## Tokenizer

ByteTokenizer maps UTF-8 bytes to IDs 0–255 and reserves 256, 257, and 258 for
PAD, BOS, and EOS. Each exported record is wrapped with BOS and EOS.
