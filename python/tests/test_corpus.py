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
            stats = tokenize_jsonl(source, output)
            self.assertEqual(stats, {"records": 1, "tokens": 3})
            payload = output.read_bytes()
            magic, version, vocab, count = struct.unpack("<8sIIQ", payload[:24])
            self.assertEqual(magic, b"FLAWDTOK")
            self.assertEqual(version, 1)
            self.assertEqual(vocab, ByteTokenizer.VOCAB_SIZE)
            self.assertEqual(count, 3)
            token_ids = struct.unpack("<3H", payload[24:])
            self.assertEqual(list(token_ids), [72, 105, ByteTokenizer.EOS_ID])
            self.assertTrue(output.with_suffix(".tokens.json").exists())


if __name__ == "__main__":
    unittest.main()
