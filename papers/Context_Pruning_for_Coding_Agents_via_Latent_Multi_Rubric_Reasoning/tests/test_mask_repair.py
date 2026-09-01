import sys
from pathlib import Path
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.inference.ast_parser import PythonAstAnalyzer
from train.inference.mask_repair import repair_mask


class MaskRepairTest(unittest.TestCase):
    def setUp(self):
        self.analyzer = PythonAstAnalyzer(require_tree_sitter=False)

    def test_repair_adds_scope_control_and_import_lines(self):
        code = (
            "import os\n"
            "@trace\n"
            "def helper(path):\n"
            "    if path:\n"
            "        return os.path.basename(path)\n"
            "    return ''"
        )
        analysis = self.analyzer.analyze(code)

        repaired = repair_mask(
            code=code,
            kept_frags=[5],
            analysis=analysis,
            dependency_hops=1,
        )

        self.assertEqual(repaired["original_kept_frags"], [5])
        self.assertEqual(repaired["repaired_kept_frags"], [1, 2, 3, 4, 5])

        reasons_by_line = {}
        for action in repaired["repair_actions"]:
            reasons_by_line.setdefault(action["line"], set()).add(action["reason"])

        self.assertIn("dependency_import_definition", reasons_by_line[1])
        self.assertIn("enclosing_function_header", reasons_by_line[2])
        self.assertIn("enclosing_function_header", reasons_by_line[3])
        self.assertIn("enclosing_if_header", reasons_by_line[4])

    def test_repair_adds_branch_companion_header_for_else_body(self):
        code = (
            "def helper(flag):\n"
            "    if flag:\n"
            "        return 1\n"
            "    else:\n"
            "        return 2"
        )
        analysis = self.analyzer.analyze(code)

        repaired = repair_mask(
            code=code,
            kept_frags=[5],
            analysis=analysis,
            dependency_hops=1,
        )

        self.assertEqual(repaired["repaired_kept_frags"], [1, 2, 4, 5])
        reasons_by_line = {}
        for action in repaired["repair_actions"]:
            reasons_by_line.setdefault(action["line"], set()).add(action["reason"])
        self.assertIn("branch_companion_header", reasons_by_line[4])
        self.assertIn("enclosing_if_header", reasons_by_line[2])


if __name__ == "__main__":
    unittest.main()
