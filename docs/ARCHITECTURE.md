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
- Averaged gradient accumulation across multiple supervised targets, with correct handling of incomplete final groups.

## Assistant training objective

Role-tagged conversations use system/user/assistant markers. FLAWDTOK v2
stores a one-byte loss flag alongside each token. System and user tokens are
available as causal context, but only assistant content is selected as a
training/evaluation target. Plain-text records retain ordinary next-token
training. The C++ trainer accepts both masked and legacy unmasked streams. Each BOS token starts a fresh context window, preventing attention from crossing document boundaries. The training CLI validates exact file length, token IDs, loss flags, PAD exclusion, and balanced BOS/EOS record boundaries before allocating the corpus. FLAWDTOK v1 files remain supported; their BOS markers are automatically excluded from loss.

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
- Atomic token/manifest writes so malformed input cannot leave a partial token file in place; empty corpora are rejected.

## Engineering constraints

- Keep the baseline dependency-light and CPU-friendly.
- Accumulate gradients over four supervised targets per optimizer update in the CLI, averaging partial final groups correctly.
- Evaluate held-out validation loss after every epoch, retain the best checkpoint, and stop after three epochs without meaningful improvement.
- Validate tensor shapes, token IDs, masks, and dataset records.
- Keep large datasets and checkpoints outside Git.
- Track dataset provenance and usage permissions.
- Never describe this experimental model as comparable to frontier hosted models.
