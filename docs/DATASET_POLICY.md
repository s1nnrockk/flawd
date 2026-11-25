# FLAWD dataset policy

This policy applies before any real model training. The repository contains tooling and tests, not a production training corpus.

## Sources and permissions

- Use only datasets with a verified license or explicit permission for the intended training use.
- Record the source name, source description, license identifier, language, and preprocessing steps in the exported corpus manifest.
- Do not scrape private accounts, private chats, paywalled material, or personal files without clear permission.
- Exclude personal secrets, credentials, private contact details, and unnecessary sensitive personal information.
- Keep source archives and generated corpora out of Git; the repository .gitignore excludes local datasets and checkpoints.

## Dataset structure

- Prefer complete instruction/response examples with explicit system, user, and assistant roles.
- Validate role order and reject role-marker injection inside message content.
- Keep documents intact during train/validation splitting. Never split adjacent tokens from one conversation between training and validation.
- Deduplicate before splitting to reduce leakage.
- Keep a separate evaluation set that is not used for tokenizer fitting, training, or hyperparameter selection.

## Quality checks before training

1. Verify the license and provenance for every source.
2. Count records, languages, role distributions, and supervised assistant tokens.
3. Inspect representative samples for malformed encoding, duplicated boilerplate, spam, and accidental secrets.
4. Measure train/validation overlap after normalization and deduplication.
5. Run the behavior evaluation suite before and after each training stage.
6. Preserve the original source data and record the exact tokenizer fingerprint and model checkpoint.

## Instruction-tuning priorities

The first curated instruction dataset should emphasize clear helpful answers, honest uncertainty, following formatting constraints, multilingual Russian/English coverage, and safe handling of requests the model should not fulfill. Do not treat raw next-token loss as proof of instruction-following or safety.
