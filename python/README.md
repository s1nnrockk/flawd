# FLAWD Python

Python contains tokenizer experiments and data preparation utilities. The
numerical engine and neural-network primitives are implemented in C++.

## Requirements

Python 3.10 or newer. The baseline package has no runtime dependencies.

## Tests

From this directory:

```powershell
python -m unittest discover -s tests -v
```

## Dataset format

Use UTF-8 JSON Lines with one object per line:

```json
{"text":"A training passage with documented usage rights."}
{"text":"Another passage."}
```

## Clean and deduplicate

From this directory:

```powershell
python -m flawd.dataset_cli raw.jsonl clean.jsonl
```

The command validates each record, normalizes whitespace, removes exact
duplicates, and prints input/output counts. Keep source records and licensing
notes alongside private datasets outside the public repository.

## Tokenizer

The current ByteTokenizer maps UTF-8 bytes directly to IDs 0–255 and reserves
256, 257, and 258 for PAD, BOS, and EOS. It is a transparent baseline; a
subword tokenizer can replace it when the model architecture is ready.
