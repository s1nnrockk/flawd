# FLAWD architecture

FLAWD is built in layers, with the model runtime preceding any public website or
subscription system. The current implementation is a small educational and
experimental foundation, not yet a complete language model.

## C++ runtime

- Tensor: contiguous float storage, shape validation, elementwise operations,
  matrix multiplication, transpose, ReLU, and stable softmax.
- Scalar autograd: reverse-mode differentiation over a computation graph.
- Neural layers: Linear, Embedding, ReLU, Tanh, LayerNorm, and Sequential.
- Optimization: mean-squared-error loss and stochastic gradient descent.

## Python utilities

- ByteTokenizer: lossless UTF-8 byte encoding with reserved PAD, BOS, and EOS IDs.
- Dataset pipeline: JSONL validation, whitespace normalization, exact deduplication.
- Dataset CLI: converts a raw JSONL file into a cleaned JSONL file.

## Planned language-model path

1. Token IDs are converted to token embeddings.
2. Positional information is added to token representations.
3. Transformer blocks apply normalization, causal self-attention, and feed-forward layers.
4. A vocabulary projection produces next-token logits.
5. Softmax and cross-entropy provide the training objective.
6. The training loop updates parameters and writes checkpoints.
7. The inference runtime generates tokens from trained checkpoints.

## Engineering constraints

- Keep the first implementation dependency-light.
- Validate shapes, token IDs, and dataset records at module boundaries.
- Avoid graph ownership cycles in autograd.
- Keep datasets and checkpoints outside Git unless intentionally publishing small samples.
- Record dataset provenance and usage permissions.
- Measure memory and throughput before implementing more complex acceleration.

## Delivery order

1. Numerical engine and tests.
2. Autograd and neural-network primitives.
3. Transformer architecture and model configuration.
4. Training, checkpointing, and evaluation.
5. Local inference.
6. Hosted API and website.
