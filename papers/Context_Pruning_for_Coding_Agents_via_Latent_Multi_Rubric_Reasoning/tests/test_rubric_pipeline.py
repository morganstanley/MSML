import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.core.rubric import RUBRIC_DIMENSIONS, line_spans_for_code, validate_rubric_item
from train.inference.ast_parser import PythonAstAnalyzer
from train.inference.build_rubric_label import enrich_item


class RubricPipelineTest(unittest.TestCase):
    def test_rubric_vectors_align_with_lines_and_spans(self):
        code = (
            "import os\n"
            "\n"
            "class PathHelper:\n"
            "    def basename(self, path):\n"
            "        return os.path.basename(path)"
        )
        item = {
            "query": "Where is the basename returned?",
            "code": code,
            "score": 0.9,
            "kept_frags": [5],
        }

        enriched = enrich_item(
            item,
            analyzer=PythonAstAnalyzer(require_tree_sitter=False),
            include_structural_metadata=True,
        )

        spans = line_spans_for_code(code)
        self.assertEqual(enriched["rubric_schema"], RUBRIC_DIMENSIONS)
        self.assertEqual(len(enriched["rubric_scores"]), len(spans))
        self.assertEqual(enriched["line_spans"], spans)
        validate_rubric_item(enriched)

        for span, original_line in zip(spans, code.split("\n")):
            self.assertEqual(code[span[0] : span[1]], original_line)

        semantic_idx = RUBRIC_DIMENSIONS.index("semantic")
        syntax_idx = RUBRIC_DIMENSIONS.index("syntax")
        dependency_idx = RUBRIC_DIMENSIONS.index("dependency")
        context_idx = RUBRIC_DIMENSIONS.index("context")

        self.assertEqual(enriched["rubric_scores"][4][semantic_idx], 0.9)
        self.assertEqual(enriched["rubric_scores"][0][semantic_idx], 0.0)

        self.assertGreaterEqual(enriched["rubric_scores"][2][syntax_idx], 1.0)
        self.assertGreaterEqual(enriched["rubric_scores"][3][syntax_idx], 1.0)

        self.assertGreater(enriched["rubric_scores"][0][dependency_idx], 0.0)
        self.assertGreaterEqual(enriched["rubric_scores"][2][context_idx], 1.0)
        self.assertGreaterEqual(enriched["rubric_scores"][3][context_idx], 1.0)

    def test_invalid_rubric_length_is_rejected(self):
        item = {
            "code": "x = 1\ny = x",
            "rubric_scores": [[1.0, 0.0, 0.0, 0.0]],
        }

        with self.assertRaises(ValueError):
            validate_rubric_item(item)

    def test_rubric_prefers_final_kept_frags_from_repaired_dataset(self):
        code = "a = 1\nb = a\nprint(b)"
        item = {
            "query": "Where is b printed?",
            "code": code,
            "score": 0.8,
            "kept_frags": [2],
            "final_kept_frags": [3],
            "accepted": True,
        }

        enriched = enrich_item(
            item,
            analyzer=PythonAstAnalyzer(require_tree_sitter=False),
            include_structural_metadata=False,
        )

        semantic_idx = RUBRIC_DIMENSIONS.index("semantic")
        self.assertEqual(enriched["rubric_parent_dataset"], "v2-repaired")
        self.assertEqual(enriched["rubric_semantic_source"], "final_kept_frags")
        self.assertEqual(enriched["rubric_scores"][2][semantic_idx], 0.8)
        self.assertEqual(enriched["rubric_scores"][1][semantic_idx], 0.0)


if __name__ == "__main__":
    unittest.main()
