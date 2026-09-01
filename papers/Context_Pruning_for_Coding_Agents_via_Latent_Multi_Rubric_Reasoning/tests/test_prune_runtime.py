import ast
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "swe-pruner" / "src"))

from swe_pruner.prune_wrapper import (
    aggregate_objective_token_logits,
    apply_ast_constraints,
    decode_code_keep_scores,
    prune_code_lines,
)
from swe_pruner.model_structure import CRFLayer
from swe_pruner.swepruner import SwePrunerOutput
from swe_pruner.runtime_ast import RuntimePythonAstAnalyzer


class PruneRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.analyzer = RuntimePythonAstAnalyzer()

    def test_multi_objective_aggregation_uses_gate_weights(self):
        rubric_token_logits = torch.tensor(
            [[[1.0, 0.0, -1.0], [0.2, 0.4, 0.8]]], dtype=torch.float32
        )
        gating_weights = torch.tensor(
            [[[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]]], dtype=torch.float32
        )

        aggregated = aggregate_objective_token_logits(
            token_logits=None,
            rubric_token_logits=rubric_token_logits,
            gating_weights=gating_weights,
            gating_type="softmax",
        )

        self.assertEqual(tuple(aggregated.shape), (1, 2))
        self.assertAlmostEqual(float(aggregated[0, 0]), -0.3, places=6)
        self.assertAlmostEqual(float(aggregated[0, 1]), 0.66, places=6)

    def test_decode_code_keep_scores_uses_final_crf(self):
        final_crf = CRFLayer(num_tags=2)
        with torch.no_grad():
            final_crf.start_transitions.zero_()
            final_crf.end_transitions.zero_()
            final_crf.transitions.zero_()

        wrapped_model = SimpleNamespace(
            model=SimpleNamespace(
                compression_head_type="crf",
                num_objectives=3,
                use_final_crf=True,
                final_crf=final_crf,
                compression_head=SimpleNamespace(crf_layers=[CRFLayer(num_tags=2)]),
            )
        )
        outputs = SwePrunerOutput(
            compression_emissions=torch.zeros((1, 4, 3, 2), dtype=torch.float32),
            fused_emissions=torch.tensor(
                [
                    [
                        [0.0, 0.0],
                        [0.0, 3.0],
                        [0.0, 2.0],
                        [0.0, 0.0],
                    ]
                ],
                dtype=torch.float32,
            ),
        )

        decoded = decode_code_keep_scores(
            wrapped_model,
            outputs,
            doc_start=1,
            doc_end=3,
        )

        self.assertEqual(decoded.tolist(), [[1.0, 1.0]])

    def test_ast_constraints_add_scope_dependency_and_return_lines(self):
        code = (
            "import os\n"
            "def helper(path):\n"
            "    base = os.path.basename(path)\n"
            "    return base"
        )
        ast_graph = self.analyzer.analyze(code)

        repaired = apply_ast_constraints(
            code,
            initial_kept_lines=[3],
            ast_graph=ast_graph,
            dependency_hops=1,
        )

        self.assertEqual(repaired, [1, 2, 3, 4])

    def test_ast_constraints_add_multiline_bracket_closure(self):
        code = (
            "def build():\n"
            "    data = [\n"
            "        'a',\n"
            "        'b',\n"
            "    ]\n"
            "    return data"
        )
        ast_graph = self.analyzer.analyze(code)

        repaired = apply_ast_constraints(
            code,
            initial_kept_lines=[3],
            ast_graph=ast_graph,
            dependency_hops=1,
        )

        self.assertEqual(repaired, [1, 2, 3, 5, 6])

    def test_prune_code_lines_applies_graph_repair(self):
        code = (
            "import os\n"
            "def helper(path):\n"
            "    base = os.path.basename(path)\n"
            "    return base\n"
            "value = 1"
        )
        ast_graph = self.analyzer.analyze(code)

        pruned_code, kept_frags = prune_code_lines(
            code=code,
            line_scores={3: 0.95},
            threshold=0.5,
            ast_graph=ast_graph,
        )

        self.assertEqual(kept_frags, [1, 2, 3, 4])
        self.assertIn("import os", pruned_code)
        self.assertIn("def helper(path):", pruned_code)
        self.assertIn("return base", pruned_code)
        self.assertNotIn("value = 1", pruned_code)
        ast.parse(pruned_code)

    def test_filtered_placeholders_are_indentation_safe_python_comments(self):
        code = (
            "def helper(value):\n"
            "    first = value + 1\n"
            "    second = value + 2\n"
            "    return first"
        )
        ast_graph = self.analyzer.analyze(code)

        pruned_code, kept_frags = prune_code_lines(
            code=code,
            line_scores={2: 0.95, 4: 0.95},
            threshold=0.5,
            ast_graph=ast_graph,
        )

        self.assertEqual(kept_frags, [1, 2, 4])
        self.assertIn("    # (filtered 1 lines)", pruned_code)
        ast.parse(pruned_code)

    def test_empty_suite_uses_pass_placeholder(self):
        code = (
            "def helper(value):\n"
            "    first = value + 1\n"
            "result = 1"
        )

        pruned_code, kept_frags = prune_code_lines(
            code=code,
            line_scores={1: 0.95, 3: 0.95},
            threshold=0.5,
            ast_graph=None,
        )

        self.assertEqual(kept_frags, [1, 3])
        self.assertIn("    pass  # (filtered 1 lines)", pruned_code)
        ast.parse(pruned_code)

    def test_ast_constraints_keep_full_multiline_string_span(self):
        code = (
            '"""Module doc line 1\n'
            'Module doc line 2\n'
            'Module doc line 3"""\n'
            "from x import y\n"
            "value = 1"
        )
        ast_graph = self.analyzer.analyze(code)

        repaired = apply_ast_constraints(
            code,
            initial_kept_lines=[2, 4],
            ast_graph=ast_graph,
            dependency_hops=0,
        )

        self.assertEqual(repaired, [1, 2, 3, 4])
        pruned_code, _ = prune_code_lines(
            code=code,
            line_scores={line: 1.0 for line in repaired},
            threshold=0.5,
            ast_graph=None,
        )
        ast.parse(pruned_code)

    def test_multiline_header_suite_placeholder_uses_block_indent(self):
        code = (
            "def merge_dictionaries(dict1,\n"
            "                       dict2):\n"
            "    first = 1\n"
            "    return dict1\n"
        )

        pruned_code, kept_frags = prune_code_lines(
            code=code,
            line_scores={1: 0.95, 2: 0.95, 4: 0.95},
            threshold=0.5,
            ast_graph=self.analyzer.analyze(code),
        )

        self.assertEqual(kept_frags, [1, 2, 4])
        self.assertIn("    pass  # (filtered 1 lines)", pruned_code)
        self.assertNotIn("                       pass", pruned_code)
        ast.parse(pruned_code)


if __name__ == "__main__":
    unittest.main()
