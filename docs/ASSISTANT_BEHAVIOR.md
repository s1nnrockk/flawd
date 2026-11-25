# FLAWD assistant behavior goals

FLAWD uses Claude as a product-quality reference: calm, natural conversation;
careful reasoning; strong coding help; useful long-context work; and honest
communication about uncertainty. This is a design reference, not a claim that
the current experimental model matches Claude or uses Anthropic weights,
datasets, prompts, or proprietary implementation.

## Target interaction qualities

1. **Helpful:** understand the actual request, complete the useful part, and
   avoid unnecessary follow-up questions.
2. **Honest:** distinguish verified facts, estimates, and unknowns. Never claim
   to have run code, opened a file, or completed an action unless it happened.
3. **Careful:** consider safety, privacy, and potential misuse before answering.
4. **Clear:** explain complex work in a structured way without padding or
   pretending certainty.
5. **Capable with code:** produce complete, coherent changes; reason about
   edge cases; include tests; and explain how to run the result.
6. **Context-aware:** use conversation history when relevant, while avoiding
   unsupported assumptions and respecting requests to avoid a topic.
7. **Multilingual:** support Russian and English well first, then expand with
   measured evaluation across additional languages.

## Model-development implications

The current byte-level Transformer is only a tiny next-token prediction
prototype. These behavior goals require much more than increasing parameter
count. The planned path is:

1. Improve the tokenizer, context handling, batching, and model efficiency;
   configurable multi-head causal attention is now part of the tiny baseline.
2. Build a legally documented, high-quality pretraining corpus with language
   balance, deduplication, and held-out evaluation data.
3. Assistant-only loss masking is implemented in FLAWDTOK v2 and the C++ trainer.
   Next, build supervised instruction-tuning datasets with carefully reviewed
   examples covering conversation, explanations, coding, tool use, and multilingual tasks.
4. Add preference training only after a reliable supervised baseline exists;
   use clear evaluation criteria and avoid optimizing for flattery or verbosity.
5. Evaluate factuality, calibration, coding correctness, instruction following,
   context retention, multilingual quality, and safety before each release.
6. Keep model quality claims tied to reproducible benchmark results.

## Dataset requirements

- Track source, license/permission, language, collection date, and processing
  steps for each dataset.
- Keep validation and test examples separate from training data.
- Remove exact and near duplicates to reduce memorization and leakage.
- Do not ingest private conversations, credentials, or personal records without
  explicit rights and appropriate consent.
- Include examples where the correct answer is to say "I don't know", ask one
  necessary clarification, or decline unsafe assistance.

## Current limitations

The present FLAWD model has a byte-level vocabulary, a very short context window,
a compact multi-head architecture, and a basic next-token trainer. Conversation serialization and assistant-only loss masking are implemented.
A sufficiently large, reviewed supervised instruction-tuning run has not yet been performed. It is an engineering
prototype, not a Claude-level assistant. These goals describe the direction;
they are not current capabilities.
