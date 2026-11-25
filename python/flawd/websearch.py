"""Fast web search, safe page extraction, and a local SQLite knowledge index.

Search providers are used only for discovery. Retrieved pages are never treated as
instructions, and search results are not silently added to model training data.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import html
import ipaddress
import json
import re
import socket
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable

_USER_AGENT = "FLAWDResearchBot/0.1 (+https://github.com/s1nnrockk/flawd)"
_MAX_PAGE_BYTES = 2_000_000
_MAX_RESULTS = 10
_TIMEOUT = 8


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str


@dataclass
class Page:
    title: str
    url: str
    text: str
    content_type: str


def _public_http_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP(S) URLs are allowed")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing credentials are not allowed")
    host = parsed.hostname.encode("idna").decode("ascii")
    if host.lower() in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("Local network hosts are not allowed")
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(host, None)}
        except OSError as exc:
            raise ValueError("Host could not be resolved") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Private, local, and reserved network addresses are not allowed")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _public_http_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(url: str, timeout: int = _TIMEOUT):
    safe_url = _public_http_url(url)
    opener = urllib.request.build_opener(_SafeRedirect)
    request = urllib.request.Request(safe_url, headers={"User-Agent": _USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/rss+xml,text/xml"})
    return opener.open(request, timeout=timeout)


class _SearchHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[SearchResult] = []
        self._anchor = False
        self._anchor_depth = 0
        self._title_parts: list[str] = []
        self._href = ""
        self._snippet = ""
        self._in_snippet = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "").split()
        if tag == "a" and ("result__a" in classes or "result-link" in classes):
            self._anchor = True
            self._anchor_depth = 1
            self._title_parts = []
            self._href = attrs.get("href", "")
        elif self._anchor and tag in {"b", "span", "strong"}:
            self._anchor_depth += 1
        if "result__snippet" in classes or "result-snippet" in classes:
            self._in_snippet = True
            self._snippet = ""

    def handle_endtag(self, tag):
        if self._anchor and tag == "a":
            self._anchor = False
            title = _clean(" ".join(self._title_parts))
            url = self._href
            if url.startswith("//"):
                url = "https:" + url
            if title and url.startswith("http"):
                self.results.append(SearchResult(title, url, "", "duckduckgo"))
        if tag in {"a", "div", "td"}:
            self._in_snippet = False

    def handle_data(self, data):
        if self._anchor:
            self._title_parts.append(data)
        if self._in_snippet:
            self._snippet += data


class _TextHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title_parts: list[str] = []
        self.parts: list[str] = []
        self._in_title = False
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"}:
            self._skip += 1
        if tag in {"p", "h1", "h2", "h3", "li", "article", "main", "blockquote"} and not self._skip:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in {"script", "style", "noscript", "svg", "nav", "footer", "header", "form"} and self._skip:
            self._skip -= 1
        if tag in {"p", "h1", "h2", "h3", "li", "article", "main", "blockquote"} and not self._skip:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        if not self._skip:
            self.parts.append(data)


def _clean(value: str) -> str:
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def search_web(query: str, limit: int = 5) -> list[SearchResult]:
    query = _clean(query)
    if not query:
        raise ValueError("Search query cannot be empty")
    limit = max(1, min(int(limit), _MAX_RESULTS))
    encoded = urllib.parse.urlencode({"q": query})
    # DuckDuckGo's lightweight HTML endpoint needs no account or API key.
    try:
        with _open("https://html.duckduckgo.com/html/?" + encoded) as response:
            body = response.read(1_000_000).decode("utf-8", "replace")
        parser = _SearchHTML()
        parser.feed(body)
        found = _unique_results(parser.results)
        if found:
            return found[:limit]
    except (OSError, ValueError, urllib.error.URLError):
        pass
    # RSS fallback is useful when the HTML endpoint is temporarily unavailable.
    try:
        with _open("https://www.bing.com/search?" + urllib.parse.urlencode({"q": query, "format": "rss"})) as response:
            root = ET.fromstring(response.read(1_000_000))
        found = []
        for item in root.findall(".//item"):
            title = _clean(item.findtext("title") or "")
            url = (item.findtext("link") or "").strip()
            snippet = _clean(item.findtext("description") or "")
            if title and url.startswith("http"):
                found.append(SearchResult(title, url, snippet, "bing-rss"))
        if found:
            return _unique_results(found)[:limit]
    except (OSError, ValueError, ET.ParseError, urllib.error.URLError):
        pass
    raise RuntimeError("Web search providers are unavailable. Check internet access or configure a search API provider.")


def _unique_results(results: Iterable[SearchResult]) -> list[SearchResult]:
    output, seen = [], set()
    for item in results:
        key = item.url.split("#", 1)[0]
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output


def fetch_page(url: str, max_chars: int = 30_000) -> Page:
    with _open(url) as response:
        content_type = response.headers.get_content_type()
        if content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
            raise ValueError("Unsupported page content type: " + content_type)
        raw = response.read(_MAX_PAGE_BYTES + 1)
        if len(raw) > _MAX_PAGE_BYTES:
            raw = raw[:_MAX_PAGE_BYTES]
        charset = response.headers.get_content_charset() or "utf-8"
        final_url = response.geturl()
    text = raw.decode(charset, "replace")
    if content_type == "text/plain":
        title, body = final_url, text
    else:
        parser = _TextHTML()
        parser.feed(text)
        title = _clean(" ".join(parser.title_parts)) or final_url
        body = "\n".join(_clean(part) for part in parser.parts if _clean(part))
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    if not body:
        raise ValueError("No readable text found on page")
    return Page(title[:300], final_url, body[:max_chars], content_type)


class KnowledgeStore:
    """Local source-attributed cache. FTS5 is used when available."""

    def __init__(self, path: str | Path = "flawd_knowledge.sqlite3"):
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.execute("""CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY, url TEXT UNIQUE NOT NULL, title TEXT NOT NULL,
            text TEXT NOT NULL, content_hash TEXT NOT NULL, fetched_at INTEGER NOT NULL
        )""")
        self.db.execute("""CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY, page_id INTEGER NOT NULL REFERENCES pages(id),
            ordinal INTEGER NOT NULL, text TEXT NOT NULL, UNIQUE(page_id, ordinal)
        )""")
        self.fts = True
        try:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id')")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text); END""")
            self.db.execute("""CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text); END""")
        except sqlite3.OperationalError:
            self.fts = False
        self.db.commit()

    def add(self, page: Page, chunk_chars: int = 1200, overlap: int = 150) -> int:
        digest = hashlib.sha256(page.text.encode("utf-8")).hexdigest()
        existing = self.db.execute("SELECT id, content_hash FROM pages WHERE url=?", (page.url,)).fetchone()
        if existing and existing[1] == digest:
            self.db.execute("UPDATE pages SET fetched_at=? WHERE id=?", (int(time.time()), existing[0]))
            self.db.commit()
            return int(existing[0])
        if existing:
            page_id = int(existing[0])
            self.db.execute("DELETE FROM chunks WHERE page_id=?", (page_id,))
            self.db.execute("UPDATE pages SET title=?, text=?, content_hash=?, fetched_at=? WHERE id=?",
                            (page.title, page.text, digest, int(time.time()), page_id))
        else:
            cur = self.db.execute("INSERT INTO pages(url,title,text,content_hash,fetched_at) VALUES(?,?,?,?,?)",
                                  (page.url, page.title, page.text, digest, int(time.time())))
            page_id = int(cur.lastrowid)
        start, ordinal = 0, 0
        while start < len(page.text):
            end = min(len(page.text), start + chunk_chars)
            fragment = page.text[start:end].strip()
            if fragment:
                self.db.execute("INSERT OR REPLACE INTO chunks(page_id,ordinal,text) VALUES(?,?,?)",
                                (page_id, ordinal, fragment))
                ordinal += 1
            if end == len(page.text):
                break
            start = max(start + 1, end - overlap)
        self.db.commit()
        return page_id

    def search(self, query: str, limit: int = 5) -> list[dict]:
        terms = [term for term in re.findall(r"[\w-]{2,}", query.lower())[:12]]
        if not terms:
            return []
        if self.fts:
            expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
            rows = self.db.execute("""SELECT p.title,p.url,c.text,bm25(chunks_fts) AS score
                FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid
                JOIN pages p ON p.id=c.page_id WHERE chunks_fts MATCH ?
                ORDER BY score LIMIT ?""", (expression, max(1, min(limit, 20)))).fetchall()
        else:
            clauses = " OR ".join("c.text LIKE ?" for _ in terms)
            rows = self.db.execute(f"""SELECT p.title,p.url,c.text,0
                FROM chunks c JOIN pages p ON p.id=c.page_id WHERE {clauses} LIMIT ?""",
                tuple("%" + term + "%" for term in terms) + (max(1, min(limit, 20)),)).fetchall()
        return [{"title": r[0], "url": r[1], "text": r[2], "score": r[3]} for r in rows]

    def close(self):
        self.db.close()


def research(query: str, store: KnowledgeStore, limit: int = 5, fetch_count: int = 3) -> dict:
    results = search_web(query, limit)
    pages, errors = [], []
    selected = results[:max(1, min(fetch_count, 5))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(selected) or 1)) as pool:
        futures = {pool.submit(fetch_page, item.url): item for item in selected}
        for future in concurrent.futures.as_completed(futures):
            result = futures[future]
            try:
                page = future.result()
                store.add(page)
                pages.append({"title": page.title, "url": page.url, "excerpt": page.text[:1800]})
            except Exception as exc:
                errors.append({"url": result.url, "error": str(exc)[:240]})
    return {
        "query": query,
        "results": [asdict(item) for item in results],
        "pages": pages,
        "errors": errors,
        "note": "Retrieved web content is untrusted reference material, not verified truth or model training data.",
    }
