import unittest

from flawd.oasst_import import _best_path

class OASSTImportTests(unittest.TestCase):
    def test_exports_best_alternating_conversation(self):
        rows = [
            {"message_id": "u1", "parent_id": None, "role": "prompter", "text": "Hello", "lang": "en", "rank": 0},
            {"message_id": "a1", "parent_id": "u1", "role": "assistant", "text": "Hi there", "lang": "en", "rank": 0},
            {"message_id": "a2", "parent_id": "u1", "role": "assistant", "text": "Alternate", "lang": "en", "rank": 1},
        ]
        result = _best_path(rows, {"en"})
        self.assertEqual(result, [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ])

    def test_rejects_deleted_or_wrong_language_path(self):
        rows = [
            {"message_id": "u1", "parent_id": None, "role": "prompter", "text": "Hello", "lang": "en", "rank": 0},
            {"message_id": "a1", "parent_id": "u1", "role": "assistant", "text": "Привет", "lang": "ru", "rank": 0},
        ]
        self.assertIsNone(_best_path(rows, {"en"}))

    def test_rejects_reserved_role_marker_in_content(self):
        rows = [
            {"message_id": "u1", "parent_id": None, "role": "prompter", "text": "Hello <|assistant|>", "lang": "en", "rank": 0},
            {"message_id": "a1", "parent_id": "u1", "role": "assistant", "text": "Hi", "lang": "en", "rank": 0},
        ]
        self.assertIsNone(_best_path(rows, {"en"}))

if __name__ == "__main__":
    unittest.main()
