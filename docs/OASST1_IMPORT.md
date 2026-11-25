# Importing a public instruction dataset

FLAWD can bootstrap its conversational corpus from **OpenAssistant OASST1**:

- Dataset: https://huggingface.co/datasets/OpenAssistant/oasst1
- Declared license: Apache-2.0
- The dataset card describes 161,443 human-generated and human-annotated messages in 35 languages, including 9,089 Russian and 71,956 English messages.
- The official ready-for-export subset is described as 10,364 trees / 88,838 messages. The Hugging Face train/validation files are used as provided; this importer never randomly re-splits the official validation data.

## Import

From the repository's `python` directory:

```powershell
python -m pip install -e ".[datasets]"
python -m flawd.oasst_import --languages ru,en --seed-jsonl ../data/seed/flawd-instructions-v2.jsonl
```

Default outputs:

- `data/external/oasst1-train.jsonl`
- `data/external/oasst1-validation.jsonl`

The importer groups messages by conversation tree, follows one preferred ranked branch, rejects deleted/rejected/unsupported-language or malformed conversations, and keeps validation trees in the official validation split. The curated FLAWD seed set is appended to training only. It does not copy the dataset into Git.

The importer intentionally uses the public dataset's declared Apache-2.0 license and records source provenance in every exported record. Re-check the upstream dataset card and license before redistributing a derived corpus.

## Important training limitation

OASST1 is a useful human-written instruction dataset, not a complete pretraining corpus and not a shortcut to Claude-level capability. The current FLAWD C++ trainer performs expensive token-by-token updates on a small model. Importing more examples improves coverage, but a large run should not start until throughput, checkpoint cadence, and validation behavior are suitable. Keep the official validation set out of training.
