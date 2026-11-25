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
            self.assertEqual(stats, {"records": 1, "tokens": 4, "supervised_tokens": 3})
            payload = output.read_bytes()
            magic, version, vocab, count = struct.unpack("<8sIIQ", payload[:24])
            self.assertEqual(magic, b"FLAWDTOK")
            self.assertEqual(version, 2)
            self.assertEqual(vocab, ByteTokenizer.VOCAB_SIZE)
            self.assertEqual(count, 4)
            token_ids = [struct.unpack("<HB", payload[24 + i * 3:27 + i * 3]) for i in range(4)]
            self.assertEqual([item[0] for item in token_ids],
                             [ByteTokenizer.BOS_ID, 72, 105, ByteTokenizer.EOS_ID])
            self.assertEqual([item[1] for item in token_ids], [0, 1, 1, 1])
            manifest = json.loads(output.with_suffix(".tokens.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_description"], "Test corpus")
            self.assertEqual(manifest["license"], "CC0-1.0")
            self.assertEqual(manifest["language"], "en")

    def test_conversation_masks_only_assistant_content(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "conversation.jsonl"
            output = Path(directory) / "conversation.tokens"
            source.write_text(json.dumps({"messages": [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer"},
            ]}) + "\n", encoding="utf-8")
            tokenize_jsonl(source, output)
            payload = output.read_bytes()
            _, version, _, count = struct.unpack("<8sIIQ", payload[:24])
            self.assertEqual(version, 2)
            records = [struct.unpack("<HB", payload[24 + i * 3:27 + i * 3]) for i in range(count)]
            tokenizer = ByteTokenizer()
            ids = [item[0] for item in records]
            flags = [item[1] for item in records]
            decoded = tokenizer.decode(ids[1:-1])
            self.assertIn("<|system|>\nRules\n<|user|>\nQuestion\n<|assistant|>\nAnswer\n", decoded)
            answer_start = decoded.index("Answer")
            answer_end = answer_start + len("Answer\n")
            self.assertTrue(all(flags[1 + answer_start:1 + answer_end]))
            self.assertFalse(any(flags[1:1 + answer_start]))
            self.assertEqual(flags[-1], 1)


    def test_masks_multiple_assistant_turns_and_utf8_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "multi.jsonl"
            output = Path(directory) / "multi.tokens"
            source.write_text(json.dumps({"messages": [
                {"role": "user", "content": "Привет"},
                {"role": "assistant", "content": "Здравствуйте"},
                {"role": "user", "content": "Ещё вопрос"},
                {"role": "assistant", "content": "Ответ готов"},
            ]}, ensure_ascii=False) + "\n", encoding="utf-8")
            tokenize_jsonl(source, output)
            payload = output.read_bytes()
            _, version, _, count = struct.unpack("<8sIIQ", payload[:24])
            self.assertEqual(version, 2)
            records = [struct.unpack("<HB", payload[24 + i * 3:27 + i * 3]) for i in range(count)]
            ids = [item[0] for item in records]
            flags = [item[1] for item in records]
            decoded = ByteTokenizer().decode(ids[1:-1])
            for user_text in ("Привет", "Ещё вопрос"):
                start = decoded.index(user_text)
                self.assertFalse(any(flags[1 + start:1 + start + len(user_text.encode("utf-8"))]))
            for assistant_text in ("Здравствуйте\\n", "Ответ готов\\n"):
                start = decoded.index(assistant_text)
                end = start + len(assistant_text.encode("utf-8"))
                self.assertTrue(all(flags[1 + start:1 + end]))
            self.assertEqual(flags[-1], 1)


if __name__ == "__main__":
    unittest.main()
