# FLAWD Local Research API

Start from the repository's `python` directory:

```powershell
python -m flawd.web_api --port 8765
```

The server binds to `127.0.0.1` only. It is designed for a local browser UI and deliberately refuses public bind addresses.

## Endpoints

- `GET http://127.0.0.1:8765/health`
- `POST http://127.0.0.1:8765/api/search` with JSON `{"query":"your question","limit":5,"fetch":3}`
- `POST http://127.0.0.1:8765/api/knowledge` with JSON `{"query":"your question","limit":5}`

Search results are cached in memory for three minutes. Up to five pages are fetched concurrently and saved into the SQLite knowledge index. The knowledge endpoint searches only previously saved page chunks.

Example PowerShell request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/api/search -ContentType "application/json" -Body '{"query":"Python 3.13 new features","limit":5,"fetch":3}'
```

The API is a local retrieval backend, not an authenticated public service. Do not expose it directly to the internet. Retrieved web content remains untrusted and does not automatically modify model weights.

## Ask the local model with live web context

After building the native engine and training a checkpoint, run:

```powershell
python -m flawd.assistant_cli "What changed in Python 3.13?" --model ..\build\Release\model.flawd --chat ..\build\Release\flawd_chat.exe --tokenizer tokenizer.bpe
```

The command searches the web, indexes fetched pages, places short source excerpts into the model prompt, runs local inference, and prints source links. With `--offline`, it uses only the local SQLite index. This is a first retrieval-augmented generation (RAG) bridge; the small experimental model may still produce weak answers, and the current context window limits how much source text it can use.
