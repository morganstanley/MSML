# Agents

Here we cover **how agents are defined**; for what each agent does and where it
runs, see [Overview](01_overview.md).

An agent is one LLM-driven step — a focused role with its own prompt and a
restricted set of [Tools](05_tools.md) — that runs an iterative loop (think →
call tools → observe) until it reports back.

## Definitions

Every agent is declared by a markdown file under `src/alpha_lab/agents/registry/`,
grouped by phase or role (the out-of-pipeline `cli/interactive` REPL — the
`alpha-lab` chat command — lives here too). The files follow the
[Agents.md](https://agents.md/) / [Skills.md](https://agentskills.io/home)
frontmatter contract: `name`, `description`, `allowed-tools`, and `metadata`.
`alpha_lab.agents.load_agent("phase3/worker_implement")` reads one file into an
`AgentDefinition`.

The frontmatter's `prompt_source` decides where the runtime prompt comes from:

- **`inline`** — the file body is the prompt verbatim (Phase 0 and supervisor
  agents, whose prompts don't vary by domain).
- **`adapter:<key>`** — the prompt is rendered each turn from the active
  [Adapter](03_adapters.md), so it swaps automatically with the workspace adapter
  and picks up live `learnings.md`, `domain_knowledge.md`, and supervisor patches
  (all Phase 1/2/3 agents).

Validate definitions with `scripts/validate_agents_frontmatter.py`.

Agent metadata may set `max_turns` to a positive integer. The limit counts LLM
responses within one agent invocation; reaching it without calling
`report_to_user` is a failed invocation, not successful completion. Phase 2
agents use a 50-turn limit so a single Builder, Critic, Tester, or Supervisor
cannot keep Phase 2 running indefinitely.

## Providers

Alpha Lab supports OpenAI (default), Anthropic Claude — the latter either
**natively** (the Anthropic Messages API) or via **AWS Bedrock** — **xAI grok**,
and **local** vLLM models. Set the `provider` field in your
[Configuration](02_configuration.md) to switch — the rest of the system is
provider-agnostic, so nothing else changes.

- **`openai`** (default) — the OpenAI Responses API (`gpt-5.2` by default), with
  built-in web search via `web_search`.
- **`anthropic`** — the native Anthropic Messages API through the MS AI Gateway,
  and the recommended path for current-generation Claude (Opus 4.6/4.7/4.8,
  Sonnet 4.6). `reasoning_effort` maps straight to adaptive thinking +
  `output_config.effort`. There's no built-in web search on this route, so it's
  proxied through OpenAI (Claude decides *when* to search; GPT performs the lookup).
- **`bedrock`** — the AWS Bedrock Converse API for Claude, through the gateway.
  Same web-search proxy as `anthropic`.
- **`grok`** — xAI grok via the gateway's OpenAI-compatible route; same
  web-search proxy as `anthropic`/`bedrock`.
- **`local`** — an OpenAI-compatible vLLM endpoint (Kimi/GLM); set
  `LOCAL_BASE_URL` and a `model` whose name contains `glm` or `kimi`.

For Claude, `reasoning_effort` (`none`/`low`/`medium`/`high`) sets how much the
model reasons before answering: `none` disables extended thinking; higher tiers
allow more. Current-generation Claude decides how much to think adaptively rather
than at a fixed token budget.

All of these work on-prem via the MS AI Gateway (no API key; token/Kerberos
auth) and run without server-side conversation storage — history is tracked
locally. One caveat on `anthropic`'s native route: it uses Anthropic prompt
caching (`cache_control`, 1h TTL) to cut repeated-turn costs on long sessions.
Caching is a billing mechanism, not conversation storage — history is still
tracked and resent locally on every call — but it does ask the service to
retain the cached prompt prefix (system prompt, tools, conversation history)
server-side until the TTL expires. Deployments with a strict zero-retention
requirement can set `ANTHROPIC_PROMPT_CACHING=0` to send no cache markers at
all, trading the cost savings for zero server-side retention.

## Sandboxing

When [`bwrap`](https://github.com/containers/bubblewrap) is available, every
pipeline agent runs in a subprocess confined by bwrap; otherwise it runs
in-process. The sandbox mounts the repo and the virtualenv read-only and the
run's workspace read-write, and binds the NVIDIA devices only for agents marked
`needs_gpu`. Under the sandbox the child reaches the experiment and memory
databases through a proxy to the parent (`sandboxing/db_proxy.py`) rather than
opening them directly, so a single process owns each store.

Set `ALPHALAB_AGENT_NOSANDBOX=1` to force in-process execution.
