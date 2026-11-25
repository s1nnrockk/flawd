# FLAWD architecture

FLAWD is built from an original, inspectable C++ model runtime and Python data
tools. The hosted website and account system are deliberately postponed until
local training and inference are reliable.

## Current C++ implementation

- Tensor: contiguous float storage, shape validation, arithmetic, matrix multiply,
  transpose, ReLU, and stable softmax.
- Autograd: reverse-mode scalar computation graph with addition, subtraction,
  multiplication, division, tanh, ReLU, exp, sqrt, and log.
- Layers: Linear, Embedding, LayerNorm, ReLU, Tanh, and Sequential.
- Losses: mean squared error and numerically stable log-sum-exp cross entropy.
- Optimizers: SGD and Adam.
- Transformer: token and position embeddings, causal multi-head self-attention,
  pre-normalized residual blocks, feed-forward layers, and vocabulary projection.
- Checkpoints: versioned binary weights with model-configuration and size checks.
- Trainer: next-token prediction using cross entropy and Adam, plus held-out loss evaluation without optimizer updates.
- Inference: autoregressive generation with temperature and top-k sampling; zero temperature selects greedy decoding.
- Efficient last-token path: earlier blocks retain full causal sequences while the final block computes only the final query, preserving the training graph while avoiding unused final-position outputs.

Conversation records support system/user/assistant markers for consistent chat formatting. The current objective still trains on every serialized token; assistant-only loss masking and true supervised instruction tuning are future milestones, not implemented features.

The current Transformer supports configurable multi-head causal attention. It remains a small experimental baseline, not a production-scale architecture.

## Python data pipeline

- ByteTokenizer: UTF-8 byte IDs 0–255 and reserved PAD/BOS/EOS IDs; each document is wrapped with BOS/EOS.
- Dataset tools: JSONL validation, whitespace normalization, exact deduplication, deterministic document-level train/validation splitting, and role-tagged conversation serialization.
- Corpus export: writes the FLAWDTOK binary stream consumed by the C++ trainer.
- The token file header is little-endian: 8-byte magic, uint32 version, uint32
  vocabulary size, uint64 token count, then uint16 token IDs.

## Training flow

1. Prepare text as UTF-8 JSONL with documented data rights.
2. Validate, normalize, and deduplicate the corpus.
3. Convert records into the FLAWDTOK token-stream format.
4. Configure the compact Transformer and initialize its weights.
5. Train next-token predictions with cross entropy and Adam.
6. Save a versioned model checkpoint.
7. Add a separate inference/generation runtime and evaluate on held-out data.

## Assistant behavior direction

See [ASSISTANT_BEHAVIOR.md](ASSISTANT_BEHAVIOR.md) for FLAWD's Claude-inspired product-quality goals, dataset requirements, and staged path from pretraining to instruction tuning and evaluation. These are target qualities, not claims about the current tiny model.

## Engineering constraints

- Keep the baseline dependency-light.
- Validate tensor shapes, token IDs, and dataset records.
- Avoid ownership cycles in autograd graphs.
- Keep large datasets and checkpoints outside Git.
- Track dataset provenance and usage permissions.
- Measure memory and speed before adding GPU support or larger model sizes.
- Do not describe a tiny experimental model as comparable to frontier hosted models.

## Next milestones

1. Finish checkpoint and trainer validation.
2. Implement inference sampling and checkpoint loading in a chat-like CLI.
3. Add multi-head attention, causal masks, and stronger Transformer tests.
4. Add validation splits, training metrics, and reproducible data manifests.
5. Improve tokenizer and batching; profile memory and speed.
6. Only then consider a hosted API and website.
