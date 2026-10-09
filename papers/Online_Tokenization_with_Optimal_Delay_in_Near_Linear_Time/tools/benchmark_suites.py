"""Tokenizer-specific adapters for the shared benchmark runner."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.benchmark_common import (
    RssRun,
    array_digest,
    read_text,
    write_json_ids,
)


GIGATOKEN_VERSION = "0.10.0"


def require_gigatoken_version() -> None:
    """Reject benchmark environments using a different Gigatoken release."""
    try:
        actual = importlib.metadata.version("gigatoken")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"Gigatoken {GIGATOKEN_VERSION} is required for benchmarks"
        ) from error
    if actual != GIGATOKEN_VERSION:
        raise RuntimeError(
            f"Gigatoken {GIGATOKEN_VERSION} is required for benchmarks; "
            f"found {actual}"
        )


@dataclass(frozen=True)
class EncodingSpec:
    name: str
    tiktoken_name: str | None
    gigatoken_repo: str | None = None
    gigatoken_scheme: str | None = None
    tiktoken_url: str | None = None
    gigatoken_hf_tokenizer: tuple[str, str] | None = None


TIKTOKEN_ENCODINGS = (
    EncodingSpec("r50k", "r50k_base", gigatoken_repo="gpt2"),
    EncodingSpec(
        "p50k",
        "p50k_base",
        gigatoken_hf_tokenizer=(
            "Xenova/text-davinci-003",
            "898195e24794cd1710ea3d0d99668208d1fe2ec9",
        ),
    ),
    EncodingSpec(
        "cl100k",
        "cl100k_base",
        gigatoken_scheme="gpt4",
        tiktoken_url=(
            "https://openaipublic.blob.core.windows.net/encodings/"
            "cl100k_base.tiktoken"
        ),
    ),
    EncodingSpec(
        "o200k",
        "o200k_base",
        gigatoken_scheme="o200k",
        tiktoken_url=(
            "https://openaipublic.blob.core.windows.net/encodings/"
            "o200k_base.tiktoken"
        ),
    ),
)
TIKTOKEN_ENCODING_BY_NAME = {
    spec.name: spec for spec in TIKTOKEN_ENCODINGS
}
TIKTOKEN_ENCODING_ALIASES = {
    "r50k_base": "r50k",
    "p50k_base": "p50k",
    "cl100k_base": "cl100k",
    "o200k_base": "o200k",
}


def _tiktoken_cache_dir() -> Path:
    if "TIKTOKEN_CACHE_DIR" in os.environ:
        return Path(os.environ["TIKTOKEN_CACHE_DIR"])
    if "DATA_GYM_CACHE_DIR" in os.environ:
        return Path(os.environ["DATA_GYM_CACHE_DIR"])
    return Path(tempfile.gettempdir()) / "data-gym-cache"


class TiktokenSuite:
    suite_id = "tiktoken"
    description = (
        "Compare serial tiktoken, Gigatoken, and Hiriluk across the r50k, "
        "p50k, cl100k, and o200k encodings and paper corpora."
    )
    epilog = """\
Examples:
  python benchmarks/tiktoken_throughput_bench.py
  python benchmarks/tiktoken_throughput_bench.py --encoding r50k --dataset english
  python benchmarks/tiktoken_throughput_bench.py --tiny
  python benchmarks/tiktoken_throughput_bench.py --full
  python benchmarks/tiktoken_throughput_bench.py --dfa
"""
    item_label = "encoding"
    selector_flags = ("--encoding", "--model")
    selector_help = "encoding name or all (default: all)"
    implementations = ("tiktoken", "gigatoken", "hiriluk")
    supports_dfa = True
    force_dfa = False
    dump_help = (
        "benchmark readable JSON output for Gigatoken and Hiriluk; "
        "temporary files are deleted outside the timed region"
    )
    supports_dump = True
    throughput_exact_ids = True
    final_note = (
        "Only exact-equivalent reference rows are included; unsupported "
        "tiktoken/Gigatoken combinations are omitted. Gigatoken rows require "
        f"exactly version {GIGATOKEN_VERSION}."
    )

    def selected_items(self, requested: str) -> list[EncodingSpec]:
        if requested == "all":
            return list(TIKTOKEN_ENCODINGS)
        canonical = TIKTOKEN_ENCODING_ALIASES.get(requested, requested)
        try:
            return [TIKTOKEN_ENCODING_BY_NAME[canonical]]
        except KeyError as error:
            choices = ", ".join(("all", *TIKTOKEN_ENCODING_BY_NAME))
            raise ValueError(
                f"unknown encoding {requested!r}; expected one of {choices}"
            ) from error

    def item_named(self, name: str) -> EncodingSpec:
        try:
            return TIKTOKEN_ENCODING_BY_NAME[name]
        except KeyError as error:
            raise ValueError(f"unknown encoding: {name}") from error

    def supports(
        self,
        implementation: str,
        item: EncodingSpec,
    ) -> bool:
        if implementation == "hiriluk":
            return True
        if implementation == "tiktoken":
            return item.tiktoken_name is not None
        if implementation == "gigatoken":
            return (
                item.gigatoken_repo is not None
                or item.gigatoken_scheme is not None
                or item.gigatoken_hf_tokenizer is not None
            )
        raise ValueError(f"unknown implementation: {implementation}")

    def implementations_for(self, *, dfa: bool) -> tuple[str, ...]:
        if dfa:
            return ("tiktoken", "hiriluk")
        return ("gigatoken", "hiriluk")

    def rss_implementations_for(self, *, dfa: bool) -> tuple[str, ...]:
        # Gigatoken has no DFA mode, but its normal implementation remains a
        # useful memory baseline when measuring Hiriluk's DFA path.
        return self.implementations

    def preload(
        self,
        implementation: str,
        item: EncodingSpec,
        *,
        phase: str,
        output_path: Path | None,
    ) -> None:
        if implementation == "tiktoken":
            import numpy  # noqa: F401
            import tiktoken  # noqa: F401
        elif implementation == "gigatoken":
            require_gigatoken_version()
            import awkward  # noqa: F401
            import gigatoken  # noqa: F401
            import numpy  # noqa: F401
            if item.gigatoken_scheme is not None:
                import tiktoken  # noqa: F401
            if item.gigatoken_hf_tokenizer is not None:
                import huggingface_hub  # noqa: F401
        elif implementation == "hiriluk":
            import hiriluk  # noqa: F401

            if phase in {"throughput", "rss"} and output_path is None:
                import numpy  # noqa: F401
        else:
            raise ValueError(f"unknown implementation: {implementation}")

    def construct(
        self,
        implementation: str,
        item: EncodingSpec,
        *,
        dfa: bool,
        profile: bool,
    ) -> Any:
        if implementation == "tiktoken":
            import tiktoken

            assert item.tiktoken_name is not None
            return tiktoken.get_encoding(item.tiktoken_name)

        if implementation == "gigatoken":
            import gigatoken as gt

            if item.gigatoken_repo is not None:
                return gt.Tokenizer(item.gigatoken_repo)

            if item.gigatoken_hf_tokenizer is not None:
                from huggingface_hub import hf_hub_download

                repo, revision = item.gigatoken_hf_tokenizer
                tokenizer_json = hf_hub_download(
                    repo,
                    "tokenizer.json",
                    revision=revision,
                )
                return gt.Tokenizer(tokenizer_json)

            import tiktoken
            assert item.tiktoken_name is not None
            assert item.tiktoken_url is not None
            assert item.gigatoken_scheme is not None
            encoding = tiktoken.get_encoding(item.tiktoken_name)
            cache_path = _tiktoken_cache_dir() / hashlib.sha1(
                item.tiktoken_url.encode()
            ).hexdigest()
            if not cache_path.is_file():
                raise FileNotFoundError(
                    f"tiktoken rank cache is unavailable at {cache_path}"
                )
            return gt.Tokenizer.from_tiktoken(
                cache_path,
                pretokenizer=item.gigatoken_scheme,
                special_tokens=dict(encoding._special_tokens),
            )

        if implementation == "hiriluk":
            import hiriluk

            return hiriluk.get_chopper(
                item.name,
                gigatoken=not dfa,
                profile=profile,
            )
        raise ValueError(f"unknown implementation: {implementation}")

    def prepare_input(self, implementation: str, path: Path) -> str | Path:
        if implementation == "tiktoken":
            return read_text(path)
        if implementation in {"gigatoken", "hiriluk"}:
            return path
        raise ValueError(f"unknown implementation: {implementation}")

    def encode_throughput(
        self,
        implementation: str,
        encoder: Any,
        input_value: str | Path,
        output_path: Path | None,
    ) -> int:
        if implementation == "tiktoken":
            assert isinstance(input_value, str)
            assert output_path is None
            ids = encoder.encode_to_numpy(
                input_value,
                allowed_special="all",
            )
            return len(ids)

        if implementation == "gigatoken":
            import awkward as ak

            assert isinstance(input_value, Path)
            ids = encoder.encode_files(input_value, parallel=False)
            if output_path is None:
                return int(ak.sum(ak.num(ids)))
            flat = ak.to_numpy(ak.flatten(ids, axis=None))
            write_json_ids(flat, output_path)
            return int(flat.size)

        if implementation == "hiriluk":
            assert isinstance(input_value, Path)
            if output_path is None:
                return len(encoder.chop_file(input_value, output="array"))
            return int(
                encoder.chop_file(
                    input_value,
                    output="json",
                    dump=output_path,
                )
            )
        raise ValueError(f"unknown implementation: {implementation}")

    def encode_verification(
        self,
        implementation: str,
        encoder: Any,
        input_value: str | Path,
        output_path: Path | None,
    ) -> tuple[int, str]:
        if implementation == "tiktoken":
            assert isinstance(input_value, str)
            ids = encoder.encode_to_numpy(
                input_value,
                allowed_special="all",
            )
            return len(ids), array_digest(ids)
        if implementation == "gigatoken":
            import awkward as ak

            assert isinstance(input_value, Path)
            ids = encoder.encode_files(input_value, parallel=False)
            flat = ak.to_numpy(ak.flatten(ids, axis=None))
            return int(flat.size), array_digest(flat)
        if implementation == "hiriluk":
            assert isinstance(input_value, Path)
            ids = encoder.chop_file(input_value, output="array")
            return len(ids), array_digest(ids)
        raise ValueError(f"unknown implementation: {implementation}")

    def encode_rss(
        self,
        implementation: str,
        encoder: Any,
        input_value: str | Path,
        output_path: Path | None,
    ) -> RssRun:
        if implementation == "hiriluk":
            assert isinstance(input_value, Path)
            if output_path is None:
                retained = encoder.chop_file(input_value, output="array")
                tokens = len(retained)
                output_mode = "numpy-u32-array"
            else:
                tokens = int(
                    encoder.chop_file(
                        input_value,
                        output="json",
                        dump=output_path,
                    )
                )
                retained = None
                output_mode = "streamed-json-file"
            return RssRun(
                tokens=tokens,
                output_mode=output_mode,
                retained=retained,
            )

        if implementation == "tiktoken":
            assert isinstance(input_value, str)
            assert output_path is None
            ids = encoder.encode_to_numpy(
                input_value,
                allowed_special="all",
            )
            return RssRun(
                tokens=len(ids),
                output_mode="numpy-u32-array",
                retained=ids,
            )

        if implementation == "gigatoken":
            import awkward as ak

            assert isinstance(input_value, Path)
            ids = encoder.encode_files(input_value, parallel=False)
            tokens = int(ak.sum(ak.num(ids)))
            if output_path is not None:
                flat = ak.to_numpy(ak.flatten(ids, axis=None))
                write_json_ids(flat, output_path)
            return RssRun(
                tokens=tokens,
                output_mode=(
                    "awkward-u32-array+json-file"
                    if output_path is not None
                    else "awkward-u32-array"
                ),
                retained=ids,
            )
        raise ValueError(f"unknown implementation: {implementation}")

    def throughput_output_mode(
        self,
        implementation: str,
        output_path: Path | None,
    ) -> str:
        if output_path is not None:
            return (
                "streamed-json-file"
                if implementation == "hiriluk"
                else "awkward-u32-array+json-file"
            )
        return {
            "tiktoken": "numpy-u32-array",
            "gigatoken": "awkward-u32-array",
            "hiriluk": "numpy-u32-array",
        }[implementation]

    def mode_description(self, *, dfa: bool) -> str:
        return "DFA/lookup/MTC" if dfa else "SIMD/full-cache"

    def throughput_note(self, *, dfa: bool) -> str:
        if dfa:
            return (
                "tiktoken receives a preloaded string and returns a NumPy "
                "array. Hiriluk reads the file and streams JSON from Rust; "
                "its temporary file is deleted after timing."
            )
        return (
            "Gigatoken and Hiriluk are each measured two ways: returning a "
            "packed array, and reading the file and writing readable JSON to "
            "a temporary file that is deleted afterward."
        )

    def throughput_output_extension(
        self,
        implementation: str,
        *,
        dfa: bool,
        dump: bool,
    ) -> str | None:
        if implementation == "hiriluk" and (dfa or dump):
            return ".json"
        if implementation == "gigatoken" and dump:
            return ".json"
        return None

    def rss_output_extension(
        self,
        implementation: str,
        *,
        dfa: bool,
        dump: bool,
    ) -> str | None:
        return self.throughput_output_extension(
            implementation,
            dfa=dfa,
            dump=dump,
        )

    def rss_output_policy(self, *, dfa: bool, dump: bool) -> str:
        if dfa:
            return "tiktoken/gigatoken-array+hiriluk-json"
        if dump:
            return "tiktoken-array+gigatoken/hiriluk-json"
        return "native-packed-array"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    environment: str
    default_repo: str

    def repo_id(self) -> str:
        return os.environ.get(self.environment, self.default_repo)


HF_MODELS = (
    ModelSpec("gpt2", "MTC_HF_GPT2", "gpt2"),
    ModelSpec("roberta", "MTC_HF_ROBERTA", "roberta-base"),
    ModelSpec("starcoder", "MTC_HF_STARCODER", "bigcode/starcoder2-3b"),
    ModelSpec(
        "mistral",
        "MTC_HF_MISTRAL",
        "mistralai/Mistral-Nemo-Base-2407",
    ),
    ModelSpec(
        "llama4",
        "MTC_HF_LLAMA4",
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    ),
    ModelSpec("qwen", "MTC_HF_QWEN", "Qwen/Qwen3-8B"),
    ModelSpec("gptoss", "MTC_HF_GPTOSS", "openai/gpt-oss-20b"),
)
HF_MODEL_BY_NAME = {spec.name: spec for spec in HF_MODELS}
HF_MODEL_ALIASES = {"qwen3": "qwen"}


class HfSuite:
    suite_id = "hf"
    description = (
        "Compare Hugging Face Tokenizers with Hiriluk's streaming "
        "DFA/lookup/MTC reference engine."
    )
    epilog = """\
Examples:
  python benchmarks/hf_throughput_bench.py
  python benchmarks/hf_throughput_bench.py --model gpt2 --dataset english
  python benchmarks/hf_throughput_bench.py --tiny
  python benchmarks/hf_throughput_bench.py --full
"""
    item_label = "model"
    selector_flags = ("--model", "--encoding")
    selector_help = "Hugging Face model name or all (default: all)"
    implementations = ("huggingface", "hiriluk")
    supports_dfa = False
    force_dfa = True
    dump_help = (
        "retain the JSON token files produced by RSS workers; otherwise "
        "use temporary files and delete them"
    )
    supports_dump = False
    throughput_exact_ids = True
    final_note = (
        "Only exact-equivalent successful rows are included; unavailable or "
        "gated repositories are reported and skipped."
    )

    def selected_items(self, requested: str) -> list[ModelSpec]:
        if requested == "all":
            return list(HF_MODELS)
        canonical = HF_MODEL_ALIASES.get(requested, requested)
        try:
            return [HF_MODEL_BY_NAME[canonical]]
        except KeyError as error:
            choices = ", ".join(("all", *HF_MODEL_BY_NAME))
            raise ValueError(
                f"unknown model {requested!r}; expected one of {choices}"
            ) from error

    def item_named(self, name: str) -> ModelSpec:
        try:
            return HF_MODEL_BY_NAME[name]
        except KeyError as error:
            raise ValueError(f"unknown model: {name}") from error

    def supports(
        self,
        implementation: str,
        item: ModelSpec,
    ) -> bool:
        if implementation not in self.implementations:
            raise ValueError(f"unknown implementation: {implementation}")
        return True

    def implementations_for(self, *, dfa: bool) -> tuple[str, ...]:
        return self.implementations

    def rss_implementations_for(self, *, dfa: bool) -> tuple[str, ...]:
        return self.implementations

    def preload(
        self,
        implementation: str,
        item: ModelSpec,
        *,
        phase: str,
        output_path: Path | None,
    ) -> None:
        if implementation == "huggingface":
            import tokenizers  # noqa: F401
        elif implementation == "hiriluk":
            import hiriluk  # noqa: F401

            if phase == "throughput" and output_path is None:
                import numpy  # noqa: F401
        else:
            raise ValueError(f"unknown implementation: {implementation}")

    def construct(
        self,
        implementation: str,
        item: ModelSpec,
        *,
        dfa: bool,
        profile: bool,
    ) -> Any:
        if implementation == "huggingface":
            from tokenizers import Tokenizer

            return Tokenizer.from_pretrained(item.repo_id())
        if implementation == "hiriluk":
            import hiriluk

            return hiriluk.get_chopper(
                item.name,
                gigatoken=False,
                profile=profile,
            )
        raise ValueError(f"unknown implementation: {implementation}")

    def prepare_input(self, implementation: str, path: Path) -> str | Path:
        if implementation == "huggingface":
            return read_text(path)
        if implementation == "hiriluk":
            return path
        raise ValueError(f"unknown implementation: {implementation}")

    def encode_throughput(
        self,
        implementation: str,
        encoder: Any,
        input_value: str | Path,
        output_path: Path | None,
    ) -> int:
        if implementation == "huggingface":
            assert isinstance(input_value, str)
            assert output_path is None
            return len(
                encoder.encode(
                    input_value,
                    add_special_tokens=False,
                ).ids
            )
        if implementation == "hiriluk":
            assert isinstance(input_value, Path)
            assert output_path is not None
            return int(
                encoder.chop_file(
                    input_value,
                    output="json",
                    dump=output_path,
                )
            )
        raise ValueError(f"unknown implementation: {implementation}")

    def encode_verification(
        self,
        implementation: str,
        encoder: Any,
        input_value: str | Path,
        output_path: Path | None,
    ) -> tuple[int, str]:
        if implementation == "huggingface":
            assert isinstance(input_value, str)
            ids = encoder.encode(
                input_value,
                add_special_tokens=False,
            ).ids
        elif implementation == "hiriluk":
            assert isinstance(input_value, Path)
            ids = encoder.chop_file(input_value, output="array")
        else:
            raise ValueError(f"unknown implementation: {implementation}")
        return len(ids), array_digest(ids)

    def encode_rss(
        self,
        implementation: str,
        encoder: Any,
        input_value: str | Path,
        output_path: Path | None,
    ) -> RssRun:
        if implementation == "huggingface":
            assert isinstance(input_value, str)
            assert output_path is None
            encoding = encoder.encode(
                input_value,
                add_special_tokens=False,
            )
            ids = encoding.ids
            return RssRun(
                tokens=len(ids),
                output_mode="python-int-list",
                retained=(encoding, ids),
            )

        if implementation == "hiriluk":
            assert isinstance(input_value, Path)
            assert output_path is not None
            tokens = int(
                encoder.chop_file(
                    input_value,
                    output="json",
                    dump=output_path,
                )
            )
            return RssRun(
                tokens=tokens,
                output_mode="streamed-json-file",
                retained=None,
            )
        raise ValueError(f"unknown implementation: {implementation}")

    def throughput_output_mode(
        self,
        implementation: str,
        output_path: Path | None,
    ) -> str:
        if implementation == "huggingface":
            assert output_path is None
            return "python-int-list"
        assert output_path is not None
        return "streamed-json-file"

    def mode_description(self, *, dfa: bool) -> str:
        return "Hiriluk DFA/lookup/MTC"

    def throughput_note(self, *, dfa: bool) -> str:
        return (
            "Hugging Face receives a preloaded string and returns its native "
            "Python integer list. Hiriluk reads the file and streams JSON "
            "from Rust; its temporary file is deleted after timing."
        )

    def throughput_output_extension(
        self,
        implementation: str,
        *,
        dfa: bool,
        dump: bool,
    ) -> str | None:
        return ".json" if implementation == "hiriluk" else None

    def rss_output_extension(
        self,
        implementation: str,
        *,
        dfa: bool,
        dump: bool,
    ) -> str | None:
        return self.throughput_output_extension(
            implementation,
            dfa=dfa,
            dump=dump,
        )

    def rss_output_policy(self, *, dfa: bool, dump: bool) -> str:
        return "huggingface-array+hiriluk-json"


TIKTOKEN_SUITE = TiktokenSuite()
HF_SUITE = HfSuite()
SUITES = {
    TIKTOKEN_SUITE.suite_id: TIKTOKEN_SUITE,
    HF_SUITE.suite_id: HF_SUITE,
}
