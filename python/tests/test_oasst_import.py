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

    def test_language_metadata_comes_from_selected_branch_not_sibling_text(self):
        from flawd.oasst_import import convert_split
        rows = [
            {"message_id": "u1", "parent_id": None, "message_tree_id": "tree",
             "role": "prompter", "text": "Same text", "lang": "en", "rank": 0,
             "tree_state": "ready_for_export"},
            {"message_id": "a1", "parent_id": "u1", "message_tree_id": "tree",
             "role": "assistant", "text": "Answer", "lang": "en", "rank": 0,
             "tree_state": "ready_for_export"},
            {"message_id": "u2", "parent_id": None, "message_tree_id": "tree",
             "role": "prompter", "text": "Same text", "lang": "ru", "rank": 1,
             "tree_state": "ready_for_export"},
        ]
        records, _ = convert_split(rows, {"en", "ru"}, "train")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["languages"], ["en"])

    def test_rejects_reserved_role_marker_in_content(self):
        rows = [
            {"message_id": "u1", "parent_id": None, "role": "prompter", "text": "Hello <|assistant|>", "lang": "en", "rank": 0},
            {"message_id": "a1", "parent_id": "u1", "role": "assistant", "text": "Hi", "lang": "en", "rank": 0},
        ]
        self.assertIsNone(_best_path(rows, {"en"}))

    def test_rejects_flagged_quality_labels(self):
        from flawd.oasst_import import _usable
        row = {
            "message_id": "a1", "role": "assistant", "text": "Answer",
            "lang": "en", "review_result": True, "deleted": False,
            "labels": {"spam": {"value": 1.0}, "pii": {"value": 0.0}},
        }
        self.assertFalse(_usable(row, {"en"}))

    def test_accepts_clean_label_scores(self):
        from flawd.oasst_import import _usable
        row = {
            "message_id": "a1", "role": "assistant", "text": "Answer",
            "lang": "en", "review_result": True, "deleted": False,
            "labels": {"spam": {"value": 0.0}, "pii": {"value": 0.0}},
        }
        self.assertTrue(_usable(row, {"en"}))

    def test_reads_arrow_struct_label_columns(self):
        from flawd.oasst_import import _usable
        row = {
            "message_id": "a1", "role": "assistant", "text": "Answer",
            "lang": "en", "review_result": True, "deleted": False,
            "labels": {"name": ["spam", "pii"], "value": [0.0, 1.0], "count": [3, 3]},
        }
        self.assertFalse(_usable(row, {"en"}))

    def test_deduplication_keeps_validation_copy_only(self):
        from flawd.oasst_import import _deduplicate_splits
        record = {"messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]}
        train, validation, stats = _deduplicate_splits([record, record], [record])
        self.assertEqual(train, [])
        self.assertEqual(validation, [record])
        self.assertEqual(stats["train_duplicates_removed"], 2)

    def test_cross_split_dedup_preserves_all_source_licenses(self):
        from flawd.oasst_import import _deduplicate_splits
        train_record = {
            "messages": [
                {"role": "user", "content": "Same question"},
                {"role": "assistant", "content": "Same answer"},
            ],
            "source": "Dolly",
            "license": "CC-BY-SA-3.0",
        }
        validation_record = {
            "messages": [
                {"role": "user", "content": "Same question"},
                {"role": "assistant", "content": "Same answer"},
            ],
            "source": "OpenAssistant/oasst1",
            "license": "Apache-2.0",
        }
        train, validation, stats = _deduplicate_splits([train_record], [validation_record])
        self.assertEqual(train, [])
        self.assertEqual(validation[0]["sources"], ["OpenAssistant/oasst1", "Dolly"])
        self.assertEqual(validation[0]["licenses"], ["Apache-2.0", "CC-BY-SA-3.0"])
        self.assertEqual(stats["train_duplicates_removed"], 1)

    def test_reads_list_of_label_objects(self):
        from flawd.oasst_import import _usable
        row = {
            "message_id": "a1", "role": "assistant", "text": "Answer",
            "lang": "en", "review_result": True, "deleted": False,
            "labels": [{"name": "spam", "value": 1.0}, {"name": "pii", "value": 0.0}],
        }
        self.assertFalse(_usable(row, {"en"}))

if __name__ == "__main__":
    unittest.main()
