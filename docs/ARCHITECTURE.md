# FLAWD architecture

FLAWD is built from an original, inspectable C++ model runtime and Python data
tools. The hosted website and account system are deliberately postponed until
local training and inference are reliable.

## Current C++ implementation

- Tensor operations and reverse-mode scalar autograd.
- Linear, Embedding, LayerNorm, activation layers, stable cross entropy, SGD, and Adam.
- Transformer with token/position embeddings, causal multi-head attention,
  pre-normalized residual blocks, feed-forward layers, and vocabulary projection.
- Versioned checkpoints with model-configuration and size validation.
- Autoregressive local inference with temperature and top-k sampling.
- Next-token training and held-out evaluation, both supporting per-token loss masks.

## Assistant training objective

Role-tagged conversations use system/user/assistant markers. FLAWDTOK v2
stores a one-byte loss flag alongside each token. System and user tokens are
available as causal context, but only assistant content is selected as a
training/evaluation target. Plain-text records retain ordinary next-token
training. The C++ trainer accepts both masked and legacy unmasked streams;
FLAWDTOK v1 files are interpreted as all-target corpora.

This implements assistant-only loss masking, a necessary supervised
instruction-tuning primitive. Conversation records are structurally validated:
roles alternate, an optional system message appears only first, every message
has content, and the final turn is an assistant response. It does not by itself provide a high-quality
assistant: results still depend on sufficient licensed data, model capacity,
training duration, and rigorous evaluation.

## Python data pipeline

- UTF-8 byte tokenizer with reserved PAD/BOS/EOS IDs.
- JSONL validation, normalization, deduplication, and deterministic splits.
- Role-tagged conversation serialization with code formatting preservation.
- FLAWDTOK v2 export with assistant-only loss masks and provenance manifest.

## Engineering constraints

- Keep the baseline dependency-light and CPU-friendly.
- Validate tensor shapes, token IDs, masks, and dataset records.
- Keep large datasets and checkpoints outside Git.
- Track dataset provenance and usage permissions.
- Never describe this experimental model as comparable to frontier hosted models.
