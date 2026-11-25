# First FLAWD training run

The first training target is a compact instruction-tuning baseline, not a pretrained general-purpose model.

## Corpus

Source: `data/seed/flawd-instructions-v1.jsonl`

- 73 original Russian/English conversations.
- One system message, one user message, and one assistant response per record.
- Conversation-level deterministic split: 90% training, 10% validation.
- Exact deduplication before splitting.
- BPE vocabulary fitted only on the training split.
- Assistant-only loss mask: system and user messages are context, not targets.

## Automated run

A push touching the seed corpus, Python data pipeline, C++ engine, or this workflow starts `.github/workflows/train.yml`. It builds the native trainer on a GitHub-hosted Windows CPU runner, trains for up to two epochs, and uploads:

- `model.flawd` — best validation checkpoint.
- `tokenizer.bpe` — matching tokenizer required for inference.
- `training.log` — epoch training and validation losses.
- `RUN.txt` — model, split, and run metadata.
- token manifests for the train and validation splits.

Download the artifact from **GitHub → Actions → FLAWD seed training → latest successful run → Artifacts**. Keep the checkpoint and tokenizer together.

## Local Windows run

From the repository root, with Python 3.10+ and CMake installed:

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release --target flawd_train

cd python
New-Item -ItemType Directory -Force training-run | Out-Null
python -m flawd.dataset_cli ../data/seed/flawd-instructions-v1.jsonl training-run/train.jsonl --validation-output training-run/validation.jsonl --validation-ratio 0.10 --seed 2026
python -m flawd.tokenizer_cli training-run/train.jsonl training-run/tokenizer.bpe --vocab-size 512
python -m flawd.corpus_cli training-run/train.jsonl training-run/train.tokens --tokenizer training-run/tokenizer.bpe --license MIT --language ru,en
python -m flawd.corpus_cli training-run/validation.jsonl training-run/validation.tokens --tokenizer training-run/tokenizer.bpe --license MIT --language ru,en
..\build\Release\flawd_train.exe training-run\train.tokens training-run\model.flawd 2 training-run\validation.tokens - 64 32 64 2 4
```

The matching local inference command is:

```powershell
..\build\Release\flawd_chat.exe training-run\model.flawd 0.8 40 training-run\tokenizer.bpe
```

## Interpreting the result

A falling training loss alone is not proof of useful behavior. Compare validation loss, inspect generated answers, and retain the tokenizer that matches the checkpoint. The initial 73 examples are a smoke-training seed; meaningful quality requires a much larger, carefully reviewed and licensed corpus.

The hosted run's elapsed time is the first reliable throughput measurement for this exact model and dataset. Use its duration and supervised-token count to estimate later runs; CPU speed varies substantially between machines.
