# Evaluation

`alpha-lab-evaluate` allows for mechanical and model-as-a-judge evaluation of
[Adapter](03_adapters.md) customization results — a way to score, and compare, how
well Phase 0 tailored the adapter to a task. Evaluations are configured via YAML,
with examples present in `tests/fixtures/evaluations/`.

To execute an evaluation, the minimum requirements are a workspace with customized
adapter content and logs, and a named set of evaluation criteria from the YAML
file.

```bash
# Uses the default tests/fixtures/evaluations/adapter_evaluation.yaml
alpha-lab-evaluate --workspace workspace_llm_speedrun_phase0 --eval-name llm_speedrun_pleias
```

Output includes detailed pass/fail explanations for each criterion in the
evaluation. The table output can be disabled, which will limit output to numeric
metrics:

```bash
alpha-lab-evaluate --workspace workspace_llm_speedrun_phase0 --eval-name llm_speedrun_pleias --no-show-table
```

```
Mechanical metrics (penalty=0.25):
  input_tokens: 387524.0  [floor=500.0, ceiling=20000.0]  (above_ceiling)
  output_tokens: 11901.0  [floor=200.0, ceiling=15000.0]  (ok)
  tool_calls: 31.0  [floor=3.0, ceiling=25.0]  (above_ceiling)
  duration_seconds: 313.7  [floor=30.0, ceiling=300.0]  (above_ceiling)
  ...
  phase1 composite: 0.2000
  phase2_builder composite: 0.8000
  ...
Section composite: 0.4239
```
