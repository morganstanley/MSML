CREATE TABLE IF NOT EXISTS benchmarks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,

    data_path TEXT NOT NULL,
    description TEXT NOT NULL,
    target TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT 'openai',
    model TEXT NOT NULL DEFAULT 'gpt-5.2',
    reasoning_effort TEXT NOT NULL DEFAULT 'low',
    shell_timeout INTEGER NOT NULL DEFAULT 300,
    tool_output_max_chars INTEGER NOT NULL DEFAULT 8000,
    pipeline_json TEXT NOT NULL,

    adapter_path TEXT,
    seed_path TEXT,

    enabled INTEGER NOT NULL DEFAULT 1,
    notes TEXT NOT NULL DEFAULT '',

    created_at TEXT,
    updated_at TEXT,
    creator TEXT,
    owner TEXT
);
