# Building the FLAWD instruction corpus

FLAWD currently uses two public instruction sources with explicit reuse terms.

## Sources

### OpenAssistant OASST1
- Dataset: https://huggingface.co/datasets/OpenAssistant/oasst1
- License declared by the dataset: Apache-2.0.
- 161,443 human-generated / human-annotated messages in 35 languages.
- The ready-for-export corpus contains 10,364 conversation trees and 88,838 messages.
- The official Hugging Face train/validation split is preserved. The source includes about 9,089 Russian and 71,956 English messages.

### Databricks Dolly 15k (optional)
- Dataset: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- 15,011 human-written English instruction records.
- License: CC BY-SA 3.0. Preserve attribution and ShareAlike obligations for redistributed derivatives.
- Covers brainstorming, classification, question answering, generation, extraction, and summarization.
- Its records are deterministically split 95/5 into train/validation. Exact duplicates are removed across both splits.

## Import

From the repository's `python` directory:

```powershell
python -m pip install -e ".[datasets]"
python -m flawd.oasst_import --languages ru,en --include-dolly --seed-jsonl ../data/seed/flawd-instructions-v2.jsonl
```

Default outputs:
- `data/external/oasst1-train.jsonl`
- `data/external/oasst1-validation.jsonl`

The importer groups OASST messages by conversation tree, follows one preferred ranked branch, rejects deleted/rejected/unsupported-language or malformed conversations, filters quality labels (spam, language mismatch, PII, inappropriate content, hate speech, sexual content, and failed-task flags), and keeps official validation trees separate. Dolly's held-out subset is also kept out of training. FLAWD's curated seed set is appended to training only.

The importer writes source and license provenance into records. It does not copy datasets into Git. Re-check upstream dataset cards and license terms before redistributing a combined corpus. For a fully automated download, normalization, BPE fit, and token export, use **Actions → FLAWD corpus preparation → Run workflow**. It uploads a prepared corpus artifact and does not start model training.

## Quality and limitations

These are useful supervised instruction sources, not a complete pretraining corpus. OASST1 is from 2023 and has a relatively small Russian portion; Dolly is English-only. They should be mixed with additional clearly licensed Russian and English material, carefully deduplicated, and evaluated on a held-out set. Do not blindly oversample or train on search-engine pages: web content can be duplicated, low quality, copyrighted, or contain personal data.

## When FLAWD is ready for a real training run

Do not start a month-long run just because the download works. Start a short baseline only after:
1. The importer completes and produces non-empty train and validation files.
2. The report shows meaningful Russian and English coverage and no train/validation duplicates.
3. The tokenizer is fitted on training data only and reused unchanged for validation.
4. A short run saves and reloads a checkpoint, and validation loss is finite.
5. The trainer supports periodic checkpoints and resume without losing optimizer state.

The current C++ trainer is a small from-scratch model with slow token-by-token updates. More hours do not make it comparable to Claude. First measure tokens/second and validation behavior on the target PC; then choose a realistic training duration. The GitHub Actions runners are for CI, not month-long persistent training.
