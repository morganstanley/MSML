import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.inference.ast_parser import PythonAstAnalyzer
from train.inference.judge_filter import heuristic_judge_repair, parse_judge_output


class JudgeFilterTest(unittest.TestCase):
    def setUp(self):
        self.analyzer = PythonAstAnalyzer(require_tree_sitter=False)

    def test_parse_judge_output_with_wrapping_text(self):
        text = """
Here is the evaluation:
```json
{
  "semantic_preservation": "pass",
  "syntax_integrity": "pass",
  "dependency_completeness": "pass",
  "context_sufficiency": "pass",
  "redundancy": "acceptable",
  "overall_quality": "high",
  "accepted": true,
  "reasoning": "looks good"
}
```
""".strip()

        evaluation = parse_judge_output(text)
        self.assertTrue(evaluation.accepted)
        self.assertEqual(evaluation.overall_quality, "high")

    def test_heuristic_judge_rejects_dropped_original_lines(self):
        code = (
            "def helper(flag):\n"
            "    if flag:\n"
            "        return 1\n"
            "    return 0"
        )
        analysis = self.analyzer.analyze(code)

        evaluation = heuristic_judge_repair(
            query="Where is the true branch return?",
            code=code,
            original_kept_frags=[3],
            repaired_kept_frags=[1, 2],
            analysis=analysis,
            dependency_hops=1,
        )

        self.assertFalse(evaluation.accepted)
        self.assertEqual(evaluation.semantic_preservation, "fail")
        self.assertEqual(evaluation.overall_quality, "low")


if __name__ == "__main__":
    unittest.main()
