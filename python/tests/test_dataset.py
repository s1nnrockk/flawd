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
                json.dumps({"text": "  hello   world  "}) + "\n"
                + json.dumps({"text": "hello world"}) + "\n"
                + json.dumps({"text": "Привет"}) + "\n",
                encoding="utf-8",
            )
            stats = build_clean_dataset(source, output)
            self.assertEqual(stats["input_records"], 3)
            self.assertEqual(stats["output_records"], 2)
            self.assertEqual(stats["duplicates_removed"], 1)
            self.assertEqual(list(read_jsonl(output)), [
                {"text": "hello world"}, {"text": "Привет"}
            ])


    def test_reproducible_train_validation_split_has_no_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "raw.jsonl"
            train = root / "train.jsonl"
            validation = root / "validation.jsonl"
            source.write_text(
                "".join(json.dumps({"text": f"record {index}"}) + "\n" for index in range(20)),
                encoding="utf-8",
            )
            stats = build_clean_dataset(source, train, validation, 0.2, 77)
            train_records = list(read_jsonl(train))
            validation_records = list(read_jsonl(validation))
            self.assertEqual(stats["training_records"], 16)
            self.assertEqual(stats["validation_records"], 4)
            self.assertFalse({item["text"] for item in train_records} &
                             {item["text"] for item in validation_records})
            second_train = root / "train2.jsonl"
            second_validation = root / "validation2.jsonl"
            build_clean_dataset(source, second_train, second_validation, 0.2, 77)
            self.assertEqual(train.read_text(encoding="utf-8"), second_train.read_text(encoding="utf-8"))
            self.assertEqual(validation.read_text(encoding="utf-8"), second_validation.read_text(encoding="utf-8"))

    def test_rejects_invalid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.jsonl"
            source.write_text('{"prompt":"missing text"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                list(read_jsonl(source))


if __name__ == "__main__":
    unittest.main()
