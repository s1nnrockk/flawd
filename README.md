# FLAWD

**FLAWD** is an independent AI project built with **C++ and Python**.

The goal is to build and train our own language model, run inference locally, and later provide a hosted API with free and paid plans.

## Principles

- No third-party hosted AI APIs for model inference.
- Build the neural-network components ourselves and understand each layer.
- Start small so development is possible on modest hardware.
- Keep model inference separate from accounts, API keys, usage limits, and billing.

## Current status

🚧 Early development. The first milestone is a small C++ tensor library, followed by tests and Python training utilities.

## Repository layout

- `engine/` — C++ numerical and inference engine.
- `python/` — Python training and tokenizer tools.
- `server/` — future account and API service.
- `tests/` — automated tests.

## Build

Requirements: CMake 3.20+ and a C++17 compiler.

```powershell
cmake -S . -B build
cmake --build build --config Release
```

## Roadmap

1. Tensor basics and unit tests.
2. Automatic differentiation.
3. Tokenizer and dataset pipeline.
4. Small Transformer model.
5. Training and evaluation.
6. Local inference.
7. Hosted API, accounts, usage limits, and subscription plans.

## Security

Never commit API secrets, passwords, private datasets, or user data.
