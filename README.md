# FLAWD

**FLAWD** is an independent language-model project built with C++ and Python.
The goal is to train an original model, run inference locally, and build a hosted
API and website only after the model runtime is dependable.

## Implemented foundation

### C++ engine
- Tensor storage, shape validation, elementwise math, matrix multiplication, transpose, ReLU, and stable softmax.
- Reverse-mode scalar automatic differentiation.
- Linear and token Embedding layers.
- ReLU, Tanh, LayerNorm, Sequential, and differentiable vector softmax.
- Numerically stable log-sum-exp cross-entropy and MSE.
- SGD and Adam optimizers.
- Configurable multi-head causal self-attention, pre-normalized Transformer blocks, and a compact byte-level Transformer model.
- Model checkpoint save/load with version and configuration validation.
- Next-token training loop and standalone `flawd_train` executable.
- Local checkpoint loading and greedy text-generation CLI (`flawd_chat`).

### Python tools
- Lossless UTF-8 byte tokenizer (IDs 0–255; PAD/BOS/EOS reserved).
- JSONL validation, whitespace normalization, and exact deduplication.
- Export from cleaned JSONL into the shared binary `.tokens` format.
- Unit tests for tokenizer, datasets, and token-file format.

### Automated checks
GitHub Actions builds the C++ targets and runs CTest and Python unit tests on Windows.

## Quick start: prepare a tiny training corpus

Create a UTF-8 JSONL file with one text record per line:

```json
{"text":"The first short training passage."}
{"text":"A second passage for the tiny experiment."}
```

Clean and tokenize it:

```powershell
cd python
python -m flawd.dataset_cli raw.jsonl train.jsonl --validation-output validation.jsonl
python -m flawd.corpus_cli train.jsonl train.tokens
python -m flawd.corpus_cli validation.jsonl validation.tokens
cd ..
```

Build FLAWD:

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Train a small model and save its checkpoint:

```powershell
.\build\Release\flawd_train.exe .\python\train.tokens .\model.flawd 1 .\python\validation.tokens
```

The initial configuration is intentionally tiny and CPU-oriented. The CLI reports held-out validation loss when a separate validation token file is supplied. After training,
start the local text-generation CLI:

```powershell
.\build\Release\flawd_chat.exe .\model.flawd
```

It accepts a prompt in the terminal. The model is an experimental byte-level
baseline, not a ChatGPT/Claude-scale model. Useful language models require much
larger licensed datasets, compute, and further engineering.

## Repository layout

- engine/ — C++ tensor engine, autograd, neural layers, Transformer, trainer.
- python/flawd/ — tokenizer and dataset/corpus utilities.
- python/tests/ — Python unit tests.
- tests/cpp/ — C++ unit tests.
- data/ — dataset format notes.
- docs/ — architecture notes.
- .github/workflows/ — CI.

## Data and security

Use datasets with documented sources and usage permissions. Never commit
private datasets, personal data, passwords, private keys, or API secrets.
