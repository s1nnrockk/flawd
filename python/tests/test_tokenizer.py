import unittest
from flawd.tokenizer import ByteTokenizer

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

    def test_invalid_id(self):
        with self.assertRaises(ValueError):
            ByteTokenizer().decode([999])

if __name__ == "__main__":
    unittest.main()
