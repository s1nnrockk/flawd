import tempfile
import unittest
from pathlib import Path

from flawd.tokenizer import ByteBPETokenizer, ByteTokenizer

class ByteTokenizerTests(unittest.TestCase):
    def test_round_trip_unicode(self):
        tokenizer = ByteTokenizer()
        text = "Привет, FLAWD! 🧠"
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_special_tokens(self):
        tokenizer = ByteTokenizer()
        ids = tokenizer.encode("hi", add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tokenizer.BOS_ID)
        self.assertEqual(ids[-1], tokenizer.EOS_ID)
        self.assertEqual(tokenizer.decode(ids), "hi")

    def test_bpe_training_is_deterministic_and_lossless(self):
        texts = ["hello hello", "hello world", "hello there"] * 4
        first = ByteBPETokenizer.train(texts, vocab_size=280, min_frequency=2)
        second = ByteBPETokenizer.train(texts, vocab_size=280, min_frequency=2)
        self.assertEqual(first.merges, second.merges)
        sample = "Привет, FLAWD! 🧠 hello hello"
        ids = first.encode(sample, add_bos=True, add_eos=True)
        self.assertEqual(ids[0], first.BOS_ID)
        self.assertEqual(ids[-1], first.EOS_ID)
        self.assertEqual(first.decode(ids), sample)
        self.assertLess(len(first.encode("hello hello")), len(ByteTokenizer().encode("hello hello")))

    def test_bpe_vocabulary_round_trip(self):
        tokenizer = ByteBPETokenizer.train(["abc abc", "abc abd"] * 5, vocab_size=270)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.bpe"
            tokenizer.save(path)
            loaded = ByteBPETokenizer.load(path)
            self.assertEqual(loaded.merges, tokenizer.merges)
            self.assertEqual(loaded.fingerprint, tokenizer.fingerprint)
            text = "abc abd — тест"
            self.assertEqual(loaded.decode(loaded.encode(text)), text)

    def test_bpe_rejects_corrupt_vocabulary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.bpe"
            path.write_bytes(b"not-a-tokenizer")
            with self.assertRaises(ValueError):
                ByteBPETokenizer.load(path)

    def test_invalid_id(self):
        with self.assertRaises(ValueError):
            ByteTokenizer().decode([999])

if __name__ == "__main__":
    unittest.main()
