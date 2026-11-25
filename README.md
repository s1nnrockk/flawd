# FLAWD

**FLAWD** is an independent AI project built with **C++ and Python**.

The goal is to build and train our own language model, run inference locally, and
only later provide a hosted API and website. The project does not depend on
third-party hosted AI APIs for its own model inference.

## Current foundation

- C++17 tensor engine with elementwise operations, matrix multiplication,
  transpose, ReLU, and softmax.
- C++ tests registered with CTest.
- Python package with a lossless UTF-8 byte tokenizer.
- JSONL dataset validation helpers.
- GitHub Actions CI for Windows C++ and Python checks.

The byte tokenizer is a simple baseline, not the final tokenizer. A trainable
subword tokenizer and the actual neural model come in later milestones.

## Repository layout

- engine/ — C++ numerical and inference engine.
- python/flawd/ — Python tokenizer and dataset utilities.
- python/tests/ — Python unit tests.
- tests/cpp/ — C++ unit tests.
- data/ — dataset format documentation; large datasets stay outside Git.
- docs/ — architecture and development notes.
- .github/workflows/ — automated checks.

## Build the C++ foundation

Requirements: CMake 3.20+ and a C++17 compiler.

```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
.\build\Release\flawd_demo.exe
```

## Run Python tests

From the repository root:

```powershell
cd python
python -m unittest discover -s tests -v
```

## Development order

1. Complete and verify the core engine.
2. Expand tokenizer and dataset tooling.
3. Add automatic differentiation and neural-network layers.
4. Implement a small Transformer.
5. Build training, checkpointing, and evaluation.
6. Add local inference and optimize it.
7. Build the hosted API and website after the model runtime is stable.

## Hardware target

The first model and tools will be designed to run in small experiments on
consumer hardware. Large-scale model training requires substantially more
memory and compute; the project will not pretend otherwise.

## Security and data

Never commit passwords, private keys, API secrets, private datasets, or user
personal data. Keep dataset source and usage permissions documented.
