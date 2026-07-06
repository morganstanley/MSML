"""Agent sandboxing: run a pipeline agent in a bwrap-confined subprocess when available.

``sandbox`` is the entry point (``run_agent``/``AgentRunHandle``);
``runner`` is the child entrypoint (``python -m alpha_lab.sandboxing.runner``); ``db_proxy``
forwards the child's store calls to the parent so it never opens a SQLite file directly.
"""
