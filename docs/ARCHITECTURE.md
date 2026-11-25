# FLAWD architecture

FLAWD is developed in layers. This milestone is a small, inspectable foundation
that runs without hosted AI APIs and without heavyweight dependencies.

## Runtime path

1. Text enters through Python data and tokenization utilities.
2. The tokenizer converts UTF-8 text into integer token IDs.
3. The future model maps token IDs to vectors and processes them with neural layers.
4. The C++ engine provides numerical operations and, later, optimized inference.
5. The output layer converts generated token IDs back into text.

## Components

- C++ engine: tensor storage and numerical primitives.
- Python package: dataset validation, tokenizer experiments, training orchestration, evaluation.
- Data format: UTF-8 JSON Lines, one object with a string "text" field per record.
- Tests: CTest for C++ and unittest for Python.

## Engineering rules

- Keep the baseline dependency-free where practical.
- Validate tensor shapes and dataset records at boundaries.
- Keep model weights and datasets out of Git unless intentionally publishing small examples.
- Never commit passwords, private keys, API secrets, or private user data.
- Measure memory and speed before adding parallelism or GPU backends.

## Planned milestones

1. Core tensor operations and tests.
2. Tokenization and dataset tooling.
3. Automatic differentiation and neural-network layers.
4. A small Transformer implementation.
5. Training, checkpointing, and evaluation.
6. Local text generation and performance work.
7. Hosted API and website after the model runtime is stable.
