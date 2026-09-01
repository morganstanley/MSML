import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.inference.row_identity import build_row_identity


class RowIdentityTest(unittest.TestCase):
    def test_same_code_different_query_have_distinct_ids(self):
        item_a = {
            "query": "Where is auth handled?",
            "code": "def helper():\n    return 1",
            "score": 0.9,
            "kept_frags": [],
        }
        item_b = {
            "query": "Refactor helper into two functions",
            "code": "def helper():\n    return 1",
            "score": 0.9,
            "kept_frags": [],
        }

        self.assertNotEqual(build_row_identity(item_a), build_row_identity(item_b))

    def test_same_code_query_different_masks_have_distinct_ids(self):
        item_a = {
            "query": "Where is helper called?",
            "code": "def helper():\n    return 1",
            "score": 0.9,
            "kept_frags": [1],
        }
        item_b = {
            "query": "Where is helper called?",
            "code": "def helper():\n    return 1",
            "score": 0.9,
            "kept_frags": [2],
        }

        self.assertNotEqual(build_row_identity(item_a), build_row_identity(item_b))

    def test_v1_input_and_v2_output_share_identity(self):
        v1_item = {
            "query": "Where is the basename call?",
            "code": "import os\n\ndef helper(path):\n    return os.path.basename(path)",
            "score": 0.93,
            "kept_frags": [4],
        }
        v2_item = {
            "query": "Where is the basename call?",
            "code": "import os\n\ndef helper(path):\n    return os.path.basename(path)",
            "score": 0.93,
            "original_kept_frags": [4],
            "repaired_kept_frags": [1, 3, 4],
            "final_kept_frags": [1, 3, 4],
            "kept_frags": [1, 3, 4],
            "accepted": True,
        }

        self.assertEqual(build_row_identity(v1_item), build_row_identity(v2_item))


if __name__ == "__main__":
    unittest.main()
