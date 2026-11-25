import json
import struct
import tempfile
import unittest
from pathlib import Path

from flawd.corpus import tokenize_jsonl
from flawd.tokenizer import ByteTokenizer


class CorpusTests(unittest.TestCase):
    def test_token_file_header_and_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            output = Path(directory) / "data.tokens"
            source.write_text(json.dumps({"text": "Hi"}) + "\n", encoding="utf-8")
            stats = tokenize_jsonl(
                source, output,
                source_description="Test corpus",
                license_id="CC0-1.0",
                language="en",
            )
            self.assertEqual(stats, {"records": 1, "tokens": 4})
            payload = output.read_bytes()
            magic, version, vocab, count = struct.unpack("<8sIIQ", payload[:24])
            self.assertEqual(magic, b"FLAWDTOK")
            self.assertEqual(version, 1)
            self.assertEqual(vocab, ByteTokenizer.VOCAB_SIZE)
            self.assertEqual(count, 4)
            token_ids = struct.unpack("<4H", payload[24:])
            self.assertEqual(list(token_ids), [ByteTokenizer.BOS_ID, 72, 105, ByteTokenizer.EOS_ID])
            manifest = json.loads(output.with_suffix(".tokens.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_description"], "Test corpus")
            self.assertEqual(manifest["license"], "CC0-1.0")
            self.assertEqual(manifest["language"], "en")


if __name__ == "__main__":
    unittest.main()
