# FLAWD data

Training corpora and model checkpoints are not committed to this repository.

The supported dataset format is UTF-8 JSON Lines (.jsonl), with one JSON object per line. Records may contain a plain string "text" field or a "messages" array with system/user/assistant roles. Conversation records must alternate user and assistant turns, may include one system message first, and must end with an assistant response.

Example record:
{"text":"A sample training passage."}

Keep private, licensed, or personally identifying data out of public repositories.
Validate datasets before training and record their source and usage permissions.

Conversation example:
{"messages":[{"role":"system","content":"Be accurate."},{"role":"user","content":"Explain a loop."},{"role":"assistant","content":"A loop repeats instructions."}]}
