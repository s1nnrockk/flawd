import json
import tempfile
import unittest
from pathlib import Path

from flawd.dataset import build_clean_dataset, read_jsonl


class DatasetTests(unittest.TestCase):
    def test_normalize_and_deduplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.jsonl"
            output = Path(directory) / "clean.jsonl"
            source.write_text(
                json.dumps({"text": "  hello   world  "}) + "\\n"
                + json.dumps({"text": "hello world"}) + "\\n"
                + json.dumps({"text": "Привет"}) + "\\n",
                encoding="utf-8",
            )
            stats = build_clean_dataset(source, output)
            self.assertEqual(stats["input_records"], 3)
            self.assertEqual(stats["output_records"], 2)
            self.assertEqual(stats["duplicates_removed"], 1)
            self.assertEqual(list(read_jsonl(output)), [
                {"text": "hello world"}, {"text": "Привет"}
            ])

    def test_rejects_invalid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.jsonl"
            source.write_text('{"prompt":"missing text"}\\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                list(read_jsonl(source))


if __name__ == "__main__":
    unittest.main()
