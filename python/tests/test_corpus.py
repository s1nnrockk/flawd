import json
import struct
import tempfile
import unittest
from pathlib import Path

from flawd.corpus import tokenize_jsonl
from flawd.tokenizer import ByteBPETokenizer, ByteTokenizer


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

    def test_empty_dataset_is_rejected_without_replacing_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.jsonl"
            output = Path(directory) / "existing.tokens"
            source.write_text("", encoding="utf-8")
            output.write_bytes(b"previous-valid-output")
            with self.assertRaisesRegex(ValueError, "empty dataset"):
                tokenize_jsonl(source, output)
            self.assertEqual(output.read_bytes(), b"previous-valid-output")
            self.assertFalse(list(Path(directory).glob("existing.tokens.*.tmp")))

    def test_invalid_input_does_not_leave_partial_token_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "broken.jsonl"
            output = Path(directory) / "data.tokens"
            source.write_text(json.dumps({"text": "valid first record"}) + "\n{broken json}\n",
                              encoding="utf-8")
            with self.assertRaises(ValueError):
                tokenize_jsonl(source, output)
            self.assertFalse(output.exists())
            self.assertFalse(list(Path(directory).glob("data.tokens.*.tmp")))

    def test_bpe_corpus_uses_trained_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.jsonl"
            tokenizer_path = Path(directory) / "tokenizer.bpe"
            output = Path(directory) / "data.tokens"
            source.write_text(json.dumps({"text": "hello hello hello"}) + "\n", encoding="utf-8")
            tokenizer = ByteBPETokenizer.train(["hello hello hello"] * 5, vocab_size=280)
            tokenizer.save(tokenizer_path)
            tokenize_jsonl(source, output, tokenizer_path=tokenizer_path)
            payload = output.read_bytes()
            magic, version, vocab, count, fingerprint = struct.unpack("<8sIIQQ", payload[:32])
            self.assertEqual(magic, b"FLAWDTOK")
            self.assertEqual(version, 3)
            self.assertEqual(vocab, tokenizer.VOCAB_SIZE)
            self.assertEqual(fingerprint, tokenizer.fingerprint)
            self.assertLess(count, len("hello hello hello".encode("utf-8")) + 2)
            records = [struct.unpack("<HB", payload[32 + i * 3:35 + i * 3]) for i in range(count)]
            self.assertEqual(tokenizer.decode([item[0] for item in records]), "hello hello hello")
            manifest = json.loads(output.with_suffix(".tokens.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["tokenizer"], tokenizer.NAME)
            self.assertEqual(manifest["vocabulary_size"], tokenizer.VOCAB_SIZE)
            self.assertEqual(manifest["tokenizer_fingerprint"], tokenizer.fingerprint)

    def test_bpe_conversation_mask_never_crosses_role_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "conversation.jsonl"
            tokenizer_path = Path(directory) / "tokenizer.bpe"
            output = Path(directory) / "conversation.tokens"
            messages = [
                {"role": "user", "content": "Привет, как дела?"},
                {"role": "assistant", "content": "Всё хорошо, спасибо!"},
            ]
            serialized = "<|user|>\nПривет, как дела?\n<|assistant|>\nВсё хорошо, спасибо!\n"
            source.write_text(json.dumps({"messages": messages}, ensure_ascii=False) + "\n",
                              encoding="utf-8")
            tokenizer = ByteBPETokenizer.train([serialized] * 8, vocab_size=320)
            tokenizer.save(tokenizer_path)
            tokenize_jsonl(source, output, tokenizer_path=tokenizer_path)
            payload = output.read_bytes()
            _, version, vocab, count, fingerprint = struct.unpack("<8sIIQQ", payload[:32])
            self.assertEqual(version, 3)
            self.assertEqual(vocab, tokenizer.VOCAB_SIZE)
            self.assertEqual(fingerprint, tokenizer.fingerprint)
            records = [struct.unpack("<HB", payload[32 + i * 3:35 + i * 3]) for i in range(count)]
            self.assertEqual(tokenizer.decode([item[0] for item in records]), serialized)
            pieces = [tokenizer.decode_bytes([item[0]]) for item in records[1:-1]]
            flags = [item[1] for item in records[1:-1]]
            decoded_offset = 0
            assistant_start = len(serialized[:serialized.index("Всё хорошо")].encode("utf-8"))
            for piece, flag in zip(pieces, flags):
                self.assertEqual(flag, 1 if decoded_offset >= assistant_start else 0)
                decoded_offset += len(piece)
            self.assertEqual(records[-1][1], 1)

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
                start = len(decoded[:decoded.index(user_text)].encode("utf-8"))
                self.assertFalse(any(flags[1 + start:1 + start + len(user_text.encode("utf-8"))]))
            for assistant_text in ("Здравствуйте\n", "Ответ готов\n"):
                start = len(decoded[:decoded.index(assistant_text)].encode("utf-8"))
                end = start + len(assistant_text.encode("utf-8"))
                self.assertTrue(all(flags[1 + start:1 + end]))
            self.assertEqual(flags[-1], 1)


if __name__ == "__main__":
    unittest.main()
