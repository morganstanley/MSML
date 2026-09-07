# Examples

Bundled configs to run as-is or copy from — the fastest way to see Alpha Lab work
on something real. Point `run.py --config` at one, or use it as a starting
template for your own [Configuration](02_configuration.md).

| Config | What it does |
|--------|--------------|
| `data/demo_exchange_config-template.json` | Synthetic FX rates, 10 experiments — quick demo |
| `data/llm_speedrun_config.json` | LLM pretraining speed/quality optimization |
| `data/paper_llm_speedrun_gpt.json` | Paper reproduction — LLM speedrun with GPT-5.2 |
| `data/paper_traffic_gpt.json` | Paper reproduction — traffic forecasting with GPT-5.2 |

The two paper reproductions also have runnable shell wrappers under `examples/`:
`run_llm_speedrun_gpt.sh` and `run_traffic_gpt.sh`.
