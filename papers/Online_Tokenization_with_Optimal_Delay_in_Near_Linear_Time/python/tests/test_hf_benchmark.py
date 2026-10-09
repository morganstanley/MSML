from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import throughput_bench as common
from tools.benchmark_suites import HF_MODEL_BY_NAME, HF_SUITE


def test_model_registry_matches_supported_hiriluk_hf_models() -> None:
    import hiriluk

    expected = [
        "gpt2",
        "roberta",
        "starcoder",
        "mistral",
        "llama4",
        "qwen",
        "gptoss",
    ]
    assert [spec.name for spec in HF_SUITE.selected_items("all")] == expected
    assert set(expected) < set(hiriluk.list_encoding_names())
    assert HF_SUITE.selected_items("qwen3")[0].name == "qwen"
    with pytest.raises(ValueError, match="unknown model"):
        HF_SUITE.selected_items("llama31p")


def test_hiriluk_constructor_selects_reference_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool, bool]] = []
    sentinel = object()

    def get_chopper(
        name: str,
        *,
        gigatoken: bool,
        profile: bool,
    ) -> object:
        calls.append((name, gigatoken, profile))
        return sentinel

    monkeypatch.setitem(
        sys.modules,
        "hiriluk",
        SimpleNamespace(get_chopper=get_chopper),
    )
    assert (
        HF_SUITE.construct(
            "hiriluk",
            HF_MODEL_BY_NAME["gpt2"],
            dfa=False,
            profile=True,
        )
        is sentinel
    )
    assert calls == [("gpt2", False, True)]


def test_hiriluk_throughput_streams_json(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    output_path = tmp_path / "tokens.json"
    path.write_text("text", encoding="utf-8")

    class Chopper:
        def __init__(self) -> None:
            self.calls: list[tuple[Path, str, Path]] = []

        def chop_file(
            self,
            file_path: Path,
            *,
            output: str,
            dump: Path,
        ) -> int:
            self.calls.append((file_path, output, dump))
            dump.write_text("[1,2,3]", encoding="utf-8")
            return 3

    chopper = Chopper()
    assert HF_SUITE.encode_throughput(
        "hiriluk",
        chopper,
        path,
        output_path,
    ) == 3
    assert chopper.calls == [(path, "json", output_path)]
    assert json.loads(output_path.read_text(encoding="utf-8")) == [1, 2, 3]


def test_huggingface_encode_disables_postprocessor_specials(
    tmp_path: Path,
) -> None:
    path = tmp_path / "input.txt"
    path.write_text("hello\n", encoding="utf-8", newline="")

    class Encoder:
        def __init__(self) -> None:
            self.calls: list[tuple[str, bool]] = []

        def encode(self, text: str, *, add_special_tokens: bool):
            self.calls.append((text, add_special_tokens))
            return SimpleNamespace(ids=[4, 5])

    encoder = Encoder()
    prepared = HF_SUITE.prepare_input("huggingface", path)
    assert prepared == "hello\n"
    assert HF_SUITE.encode_throughput(
        "huggingface",
        encoder,
        prepared,
        None,
    ) == 2
    assert encoder.calls == [("hello\n", False)]
    assert HF_SUITE.prepare_input("hiriluk", path) == path


def test_hiriluk_rss_run_streams_json(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "tokens.json"
    class Chopper:
        def chop_file(
            self,
            path: Path,
            *,
            output: str,
            dump: Path,
        ) -> int:
            assert path == input_path
            assert output == "json"
            assert dump == output_path
            dump.write_text("[1,2,3]", encoding="utf-8")
            return 3

    result = HF_SUITE.encode_rss(
        "hiriluk",
        Chopper(),
        input_path,
        output_path,
    )
    assert result.tokens == 3
    assert result.retained is None
    assert json.loads(output_path.read_text(encoding="utf-8")) == [1, 2, 3]


def test_rss_parity_rejects_same_length_different_json() -> None:
    rows = [
        {
            "impl": "huggingface",
            "encoding": "gpt2",
            "corpus": "english",
            "tokens": 3,
            "digest": "aaa",
        },
        {
            "impl": "hiriluk",
            "encoding": "gpt2",
            "corpus": "english",
            "tokens": 3,
            "digest": "bbb",
        },
    ]
    with pytest.raises(RuntimeError, match="token-ID mismatch"):
        common.validate_parity(rows, exact_ids=True)


def test_defaults_are_reference_mode_and_no_private_flags(
) -> None:
    parser = common.build_throughput_parser(HF_SUITE)
    args = parser.parse_args(())
    assert args.model == "all"
    assert args.dataset == "all"
    assert args.corpus_size == "short"
    assert not hasattr(args, "dump")
    assert not hasattr(args, "dfa")
    assert not hasattr(args, "gigatoken")
    assert parser.parse_args(("--tiny",)).corpus_size == "tiny"
    assert parser.parse_args(("--full",)).corpus_size == "full"

    for option in (
        "--warmup",
        "--reps",
        "--_worker",
        "--_impl",
        "--dump",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args((option, "value"))
