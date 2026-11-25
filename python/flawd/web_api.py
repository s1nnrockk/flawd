"""Local HTTP API for FLAWD's web research backend.

Binds to loopback by default. It is intended for a local web UI, not public hosting.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .websearch import KnowledgeStore, research

_CACHE: dict[str, tuple[float, dict]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_TTL = 180


class Handler(BaseHTTPRequestHandler):
    server_version = "FLAWDResearch/0.2"

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin", "")
        if origin in {"http://localhost", "http://127.0.0.1", "http://localhost:3000",
                      "http://127.0.0.1:3000", "http://localhost:5173",
                      "http://127.0.0.1:5173"}:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        origin = self.headers.get("Origin", "")
        allowed = {"http://localhost", "http://127.0.0.1", "http://localhost:3000",
                   "http://127.0.0.1:3000", "http://localhost:5173",
                   "http://127.0.0.1:5173"}
        if origin not in allowed:
            self._json(403, {"error": "Origin is not allowed"})
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/health":
            self._json(200, {"ok": True, "service": "flawd-research", "cache_ttl_seconds": _CACHE_TTL})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in {"/api/search", "/api/knowledge"}:
            self._json(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > 16_384:
                self._json(413, {"error": "Request body must be between 1 and 16384 bytes"})
                return
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            query = payload.get("query", "")
            if not isinstance(query, str) or not query.strip() or len(query) > 1000:
                self._json(400, {"error": "query must be a non-empty string of at most 1000 characters"})
                return
            limit = max(1, min(int(payload.get("limit", 5)), 10))
            if route == "/api/knowledge":
                with KnowledgeStore(self.server.db_path) as store:
                    result = {"query": query.strip(), "knowledge": store.search(query.strip(), limit)}
                self._json(200, result)
                return
            key = query.strip().casefold() + "|" + str(limit)
            now = time.monotonic()
            with _CACHE_LOCK:
                cached = _CACHE.get(key)
                if cached and cached[0] > now:
                    self._json(200, {**cached[1], "cached": True})
                    return
            with KnowledgeStore(self.server.db_path) as store:
                result = research(query.strip(), store, limit=limit, fetch_count=max(1, min(int(payload.get("fetch", 3)), 5)))
            with _CACHE_LOCK:
                _CACHE[key] = (now + _CACHE_TTL, result)
                for old_key, (expires, _) in list(_CACHE.items()):
                    if expires <= now:
                        _CACHE.pop(old_key, None)
            self._json(200, {**result, "cached": False})
        except json.JSONDecodeError:
            self._json(400, {"error": "Invalid JSON"})
        except (ValueError, TypeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            self._json(502, {"error": str(exc)[:400]})

    def log_message(self, fmt: str, *args) -> None:
        # Keep normal requests quiet; errors still surface through JSON responses.
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local FLAWD web research API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--db", default="flawd_knowledge.sqlite3")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("Public binding is disabled by default; use loopback for local UI access")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.db_path = args.db
    print(f"FLAWD research API: http://{args.host}:{args.port}")
    print("Endpoints: GET /health, POST /api/search, POST /api/knowledge")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping FLAWD research API...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
