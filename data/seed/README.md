# FLAWD seed corpus

## `flawd-instructions-v1.jsonl`

This is an original, manually authored bilingual seed corpus for the first FLAWD instruction-tuning experiment.

- **Records:** 73 conversations.
- **Languages:** Russian and English.
- **Format:** one JSON object per line, with system, user, and assistant messages.
- **Coverage:** programming fundamentals, web development, machine learning concepts, data handling, security hygiene, mathematics, scientific explanations, troubleshooting, uncertainty calibration, and instruction/format following.
- **Provenance:** written specifically for this repository; no third-party text or model-generated dump is included.
- **Rights:** original project-authored text, distributed under the repository's MIT license.
- **Purpose:** a small smoke-training and behavior-seeding corpus, not a foundation-model pretraining dataset.

The training workflow deduplicates and splits whole conversations deterministically, fits BPE only on the training split, exports assistant-only loss masks, trains a compact CPU-sized model, and stores the checkpoint and run metadata as an Actions artifact.

## Limitations

Seventy-three examples are not enough to teach broad language competence or approach Claude-class quality. The corpus is intentionally small and transparent so the first run can verify the end-to-end process. Future versions should expand coverage with carefully reviewed, licensed sources and larger independently authored examples. Keep private data, raw corpora, token files, and checkpoints out of Git.
