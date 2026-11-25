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
- Next-token training with averaged gradient accumulation (four supervised targets per optimizer update by default) and standalone `flawd_train` executable.
- Local checkpoint loading and autoregressive generation with temperature and top-k sampling (`flawd_chat`).

### Python tools
- Lossless UTF-8 byte tokenizer (IDs 0–255; PAD/BOS/EOS reserved).
- JSONL validation, whitespace normalization, and exact deduplication.
- Export from cleaned JSONL into the versioned binary `.tokens` format, with assistant-only loss masking for role-tagged conversations.
- The trainer uses system/user messages as context and scores assistant content only; plain text corpora retain ordinary next-token training.
- BOS boundaries isolate records so training examples never read tokens from the previous document.
- Unit tests for tokenizer, datasets, and token-file format.

### Automated checks
GitHub Actions builds the C++ targets and runs CTest and Python unit tests on Windows.

## Quick start: prepare a tiny training corpus

Create a UTF-8 JSONL file with one text record per line for ordinary language-model pretraining:

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

For a more compact vocabulary, train a byte-pair tokenizer from the training split, then use the exact same tokenizer for every split:
```powershell
cd python
python -m flawd.tokenizer_cli train.jsonl tokenizer.bpe --vocab-size 512
python -m flawd.corpus_cli train.jsonl train.tokens --tokenizer tokenizer.bpe
python -m flawd.corpus_cli validation.jsonl validation.tokens --tokenizer tokenizer.bpe
cd ..
```
The portable tokenizer file is shared by Python data preparation and native C++ inference. BPE training uses a bounded 250 KB sample and a 512-token target vocabulary by default to keep preparation manageable on CPU.

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

Model dimensions can be supplied after the optional resume checkpoint. For example, a moderate CPU experiment with a longer context is:
```powershell
.\build\Release\flawd_train.exe .\python\train.tokens .\model-128.flawd 1 .\python\validation.tokens - 128 32 64 2 4
```
The positional dimensions are context length, embedding width, feed-forward width, Transformer layer count, and attention-head count. Use `-` when no resume checkpoint is needed but dimension arguments follow. The chat program reads these dimensions from the checkpoint automatically.

For assistant-style training, use conversation records instead of plain text:

```json
{"messages":[{"role":"system","content":"Be helpful and accurate."},{"role":"user","content":"Explain what a loop does."},{"role":"assistant","content":"A loop repeats a block of instructions while a condition holds, or for each item in a sequence."}]}
```

The exporter stores assistant-only loss masks in FLAWDTOK v2. User and system messages remain visible to the model as context; assistant responses and the final end-of-turn token are training targets. Conversations must alternate user/assistant turns, may have one system message at the beginning, and must end with an assistant response. Reserved role markers inside message content are rejected.

The CLI currently accumulates gradients across four supervised targets before each Adam update; incomplete final groups are averaged correctly. This reduces optimizer-update overhead and produces a less noisy update without keeping multiple full computation graphs alive. The default configuration is intentionally tiny and CPU-oriented; larger dimensions are configurable, and inference reads the saved configuration from the checkpoint. It is suitable for verifying the end-to-end training pipeline, not for producing a capable general-purpose assistant. The CLI evaluates held-out validation loss after every epoch when a separate validation token file is supplied. With validation enabled, it saves the best checkpoint and stops after three consecutive epochs without a meaningful improvement (minimum improvement: 0.0001). This avoids blindly training for every requested epoch and keeps the best validation checkpoint rather than the final, potentially overfit epoch. Without a validation file, it runs the requested number of epochs and saves the final checkpoint. To continue from an existing checkpoint, pass the validation corpus and checkpoint as the final two arguments: `flawd_train train.tokens model.flawd 1 validation.tokens previous.flawd`. The current optimizer state is restarted when resuming; model weights are restored. After training,
start local text generation. For a byte-tokenized checkpoint:

```powershell
.\build\Release\flawd_chat.exe .\model.flawd
```

For a BPE-tokenized checkpoint, pass the matching tokenizer after temperature and top-k:

```powershell
.\build\Release\flawd_chat.exe .\model.flawd 0.8 40 .\python\tokenizer.bpe
```

It accepts a prompt in the terminal. Sampling defaults to temperature 0.8 and top-k 40; pass `0` as temperature for greedy decoding. The model is an experimental byte-level
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
