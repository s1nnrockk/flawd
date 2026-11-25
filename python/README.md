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

UTF-8 JSON Lines, one object per line:

```json
{"text":"A training passage with documented usage rights."}
{"text":"Another passage."}
```

## Prepare a corpus

```powershell
python -m flawd.dataset_cli raw.jsonl clean.jsonl
python -m flawd.corpus_cli clean.jsonl clean.tokens
```

The first command validates records, normalizes whitespace, and removes exact
duplicates. The second writes a little-endian FLAWDTOK binary token stream and
a JSON manifest. The C++ `flawd_train` executable reads this format.

## Tokenizer

ByteTokenizer maps UTF-8 bytes to IDs 0–255 and reserves 256, 257, and 258 for
PAD, BOS, and EOS. EOS is appended to each exported text record. This transparent
baseline can later be replaced by a subword tokenizer.
