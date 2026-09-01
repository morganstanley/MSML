#!/usr/bin/env python3

import sys
import types

from transformers import PreTrainedTokenizerBase


# vllm 0.6.3.post1 expects this property on some tokenizers, but the local
# Qwen tokenizer under the installed transformers build does not expose it.
if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(  # type: ignore[attr-defined]
        lambda self: list(self.all_special_tokens)
    )

# Some outlines builds import pyairports at request time. The current swe env
# has only stale package metadata and no importable module. We only need a
# process-local shim here because SWE-bench does not use airport grammars.
try:
    import pyairports.airports  # type: ignore[import-not-found]
except ModuleNotFoundError:
    pyairports_module = types.ModuleType("pyairports")
    airports_module = types.ModuleType("pyairports.airports")
    airports_module.AIRPORT_LIST = []
    pyairports_module.airports = airports_module
    sys.modules["pyairports"] = pyairports_module
    sys.modules["pyairports.airports"] = airports_module

import uvloop
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.utils import FlexibleArgumentParser


if __name__ == "__main__":
    parser = FlexibleArgumentParser(description="vLLM OpenAI-Compatible RESTful API server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    validate_parsed_serve_args(args)
    uvloop.run(run_server(args))
