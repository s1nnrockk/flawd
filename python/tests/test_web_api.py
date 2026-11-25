import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from flawd.web_api import Handler
from flawd.websearch import KnowledgeStore, Page


class WebApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp.name) / "knowledge.sqlite3")
        with KnowledgeStore(self.db_path) as store:
            store.add(Page("Test source", "https://example.com/article",
                           "A compact transformer uses attention to process context.",
                           "text/plain"))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.db_path = self.db_path
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:" + str(self.server.server_port)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temp.cleanup()

    def test_health_endpoint(self):
        with urllib.request.urlopen(self.base + "/health", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["service"], "flawd-research")

    def test_local_knowledge_endpoint(self):
        body = json.dumps({"query": "transformer attention"}).encode("utf-8")
        request = urllib.request.Request(
            self.base + "/api/knowledge", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        self.assertTrue(payload["knowledge"])
        self.assertEqual(payload["knowledge"][0]["url"], "https://example.com/article")


if __name__ == "__main__":
    unittest.main()
