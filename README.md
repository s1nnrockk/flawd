# FLAWD

**FLAWD** is an independent language-model project built with C++ and Python.
The long-term goal is to train an original model, run inference locally, and
build a hosted API and website only after the model runtime is dependable.

## Current foundation

### C++ engine
- Contiguous tensor storage and validated tensor shapes.
- Elementwise add, subtract, multiply, scaling, and ReLU.
- 2D matrix multiplication and transpose.
- Numerically stable tensor softmax.
- Scalar reverse-mode automatic differentiation.
- Trainable Linear and Embedding layers.
- ReLU, Tanh, LayerNorm, vector softmax, and Sequential.
- Mean-squared-error loss and SGD optimizer.
- CTest unit tests.

### Python tools
- Lossless UTF-8 byte tokenizer with reserved special-token IDs.
- JSONL dataset validation, whitespace normalization, and exact deduplication.
- Dataset-cleaning command line tool.
- Python unit tests.

### Automation
- GitHub Actions builds the C++ project and runs C++ and Python tests on Windows.

## Repository layout

- engine/ — C++ numerical engine, autograd, and neural-network layers.
- python/flawd/ — tokenizer and dataset utilities.
- python/tests/ — Python unit tests.
- tests/cpp/ — C++ unit tests.
- data/ — dataset format documentation.
- docs/ — architecture and engineering notes.
- .github/workflows/ — CI checks.

## Build and test

Requirements: CMake 3.20+ and a C++17 compiler.

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\flawd_demo.exe
```

Python tests:

```powershell
cd python
python -m unittest discover -s tests -v
```

Clean a dataset:

```powershell
cd python
python -m flawd.dataset_cli input.jsonl clean.jsonl
```

## Development sequence

1. Finish and verify the numerical and autograd foundations.
2. Complete Transformer primitives and model configuration.
3. Implement a small Transformer language model.
4. Add batching, training loops, checkpointing, and evaluation.
5. Add local text generation and performance work.
6. Build the hosted API, accounts, and website.

The initial implementation is intentionally small. Training a useful large
language model requires large, well-documented datasets and substantial compute;
the project will measure its limits rather than claim otherwise.

## Data and security

Training data must have documented sources and usage permissions. Never commit
private datasets, personal data, passwords, private keys, or API secrets.
