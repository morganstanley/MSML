from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tiktoken

import hiriluk
import hiriluk._hiriluk as native


TEXT = "Hello, streaming world! 👋\n<|endoftext|> café"


def reference_ids(text: str) -> list[int]:
    return tiktoken.get_encoding("r50k_base").encode(
        text, allowed_special="all"
    )


def read_u32le(path: Path) -> list[int]:
    assert path.stat().st_size % 4 == 0
    return np.fromfile(path, dtype="<u4").tolist()


def test_import_and_registry() -> None:
    assert "r50k" in hiriluk.list_encoding_names()
    assert "llama31p" not in hiriluk.list_encoding_names()
    assert hiriluk.ChopProfile is native.ChopProfile
    assert not hasattr(hiriluk, "TokenBuffer")
    assert not hasattr(native, "TokenBuffer")
    assert not hasattr(native, "_BenchmarkSession")
    assert not hasattr(native, "_BenchmarkStats")


def test_chopper_metadata() -> None:
    chopper = hiriluk.get_chopper("r50k")
    assert chopper.name == "r50k"
    assert chopper.gigatoken is False
    assert chopper.profile is False
    assert chopper.last_profile is None


def test_chop_default_and_none_outputs_return_exact_numpy_arrays() -> None:
    chopper = hiriluk.get_chopper("r50k")
    expected = reference_ids(TEXT)

    for tokens in (chopper.chop(TEXT), chopper.chop(TEXT, output=None)):
        assert isinstance(tokens, np.ndarray)
        assert tokens.dtype == np.uint32
        assert tokens.ndim == 1
        assert tokens.tolist() == expected


def test_chop_file_default_and_none_outputs_return_exact_numpy_arrays(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text(TEXT, encoding="utf-8")
    chopper = hiriluk.get_chopper("r50k")
    expected = reference_ids(TEXT)

    for tokens in (
        chopper.chop_file(input_path),
        chopper.chop_file(input_path, output=None),
    ):
        assert isinstance(tokens, np.ndarray)
        assert tokens.dtype == np.uint32
        assert tokens.ndim == 1
        assert tokens.tolist() == expected


def test_iterator_output_is_explicitly_unimplemented(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text(TEXT, encoding="utf-8")
    chopper = hiriluk.get_chopper("r50k")

    with pytest.raises(
        NotImplementedError,
        match="output='iterator' is not implemented",
    ):
        chopper.chop(TEXT, output="iterator")
    with pytest.raises(
        NotImplementedError,
        match="output='iterator' is not implemented",
    ):
        chopper.chop_file(input_path, output="iterator")


def test_gigatoken_path_matches_dfa() -> None:
    dfa = hiriluk.get_chopper("r50k").chop(TEXT, output="array")
    simd = hiriluk.get_chopper("r50k", gigatoken=True).chop(
        TEXT,
        output="array",
    )
    overridden = hiriluk.get_chopper("r50k").chop(
        TEXT,
        gigatoken=True,
        output="array",
    )
    assert simd.tolist() == dfa.tolist()
    assert overridden.tolist() == dfa.tolist()


def test_array_output_returns_one_zero_copy_numpy_u32_array() -> None:
    chopper = hiriluk.get_chopper("r50k", gigatoken=True)
    tokens = chopper.chop(TEXT, output="array")
    assert isinstance(tokens, np.ndarray)
    assert tokens.dtype == np.uint32
    assert tokens.ndim == 1
    assert tokens.flags.c_contiguous
    assert tokens.flags.writeable
    assert not tokens.flags.owndata
    assert tokens.base is not None
    assert len(tokens) == len(reference_ids(TEXT))
    assert tokens.nbytes == len(tokens) * 4
    assert tokens.tolist() == reference_ids(TEXT)
    assert int(tokens[-1]) == reference_ids(TEXT)[-1]


def test_chop_file_compact_output(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "tokens.u32le"
    input_path.write_text(TEXT, encoding="utf-8")
    chopper = hiriluk.get_chopper("r50k", gigatoken=True)

    assert chopper.chop_file(
        input_path,
        output="compact",
        dump=output_path,
    ) == len(reference_ids(TEXT))
    assert read_u32le(output_path) == reference_ids(TEXT)


def test_chop_file_array_and_profiled_outputs(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.txt"
    compact_path = tmp_path / "tokens.u32le"
    json_path = tmp_path / "tokens.json"
    input_path.write_text(TEXT, encoding="utf-8")
    chopper = hiriluk.get_chopper(
        "r50k",
        gigatoken=True,
        profile=True,
    )
    assert chopper.profile is True
    assert chopper.last_profile is None

    array = chopper.chop_file(input_path, output="array")
    assert isinstance(array, np.ndarray)
    assert array.dtype == np.uint32
    assert array.tolist() == reference_ids(TEXT)
    array_profile = chopper.last_profile
    assert isinstance(array_profile, hiriluk.ChopProfile)
    assert array_profile.total_tokens == len(reference_ids(TEXT))
    assert array_profile.ttft_seconds is not None
    assert 0.0 <= array_profile.ttft_seconds <= array_profile.elapsed_seconds

    assert chopper.chop_file(
        input_path,
        output="compact",
        dump=compact_path,
    ) == len(reference_ids(TEXT))
    compact_profile = chopper.last_profile
    assert isinstance(compact_profile, hiriluk.ChopProfile)
    assert compact_profile.total_tokens == len(reference_ids(TEXT))
    assert compact_profile.ttft_seconds is not None
    assert 0.0 <= compact_profile.ttft_seconds <= compact_profile.elapsed_seconds
    assert read_u32le(compact_path) == reference_ids(TEXT)

    assert chopper.chop_file(
        input_path,
        output="json",
        dump=json_path,
    ) == len(reference_ids(TEXT))
    json_profile = chopper.last_profile
    assert isinstance(json_profile, hiriluk.ChopProfile)
    assert json_profile.total_tokens == len(reference_ids(TEXT))
    assert json_profile.ttft_seconds is not None
    assert 0.0 <= json_profile.ttft_seconds <= json_profile.elapsed_seconds
    assert json.loads(json_path.read_text(encoding="utf-8")) == reference_ids(TEXT)


@pytest.mark.parametrize("gigatoken", [False, True])
def test_chop_profile_records_native_ttft_for_dfa_and_fast_paths(
    gigatoken: bool,
) -> None:
    chopper = hiriluk.get_chopper(
        "r50k",
        gigatoken=gigatoken,
        profile=True,
    )
    tokens = chopper.chop(TEXT, output="array")
    profile = chopper.last_profile

    assert isinstance(profile, hiriluk.ChopProfile)
    assert profile.total_tokens == len(tokens)
    assert profile.ttft_seconds is not None
    assert 0.0 <= profile.ttft_seconds <= profile.elapsed_seconds


def test_empty_chop_profile_has_no_first_token() -> None:
    chopper = hiriluk.get_chopper("r50k", gigatoken=True, profile=True)
    tokens = chopper.chop("", output="array")
    profile = chopper.last_profile

    assert len(tokens) == 0
    assert isinstance(profile, hiriluk.ChopProfile)
    assert profile.total_tokens == 0
    assert profile.ttft_seconds is None
    assert profile.elapsed_seconds >= 0.0


@pytest.mark.parametrize("output", ["json", "compact"])
def test_file_output_requires_dump(output: str) -> None:
    with pytest.raises(ValueError, match="require dump"):
        hiriluk.get_chopper("r50k").chop(TEXT, output=output)


def test_output_and_dump_combinations_are_validated(tmp_path: Path) -> None:
    output_path = tmp_path / "tokens"
    chopper = hiriluk.get_chopper("r50k")
    with pytest.raises(ValueError, match="dump requires"):
        chopper.chop(TEXT, dump=output_path)
    with pytest.raises(ValueError, match="cannot be combined"):
        chopper.chop(TEXT, output="array", dump=output_path)
    with pytest.raises(ValueError, match="unknown output"):
        chopper.chop(TEXT, output="text")
    with pytest.raises(TypeError, match="memory"):
        chopper.chop(TEXT, memory=True)


def test_chop_string_json_and_compact_outputs(tmp_path: Path) -> None:
    compact_path = tmp_path / "tokens.u32le"
    json_path = tmp_path / "tokens.json"
    chopper = hiriluk.get_chopper("r50k")

    assert chopper.chop(
        TEXT,
        output="compact",
        dump=compact_path,
    ) == len(reference_ids(TEXT))
    assert read_u32le(compact_path) == reference_ids(TEXT)

    assert chopper.chop(
        TEXT,
        output="json",
        dump=json_path,
    ) == len(reference_ids(TEXT))
    assert json.loads(json_path.read_text(encoding="utf-8")) == reference_ids(TEXT)


def test_empty_input(tmp_path: Path) -> None:
    chopper = hiriluk.get_chopper("r50k")
    tokens = chopper.chop("", output="array")
    assert isinstance(tokens, np.ndarray)
    assert tokens.dtype == np.uint32
    assert tokens.tolist() == []
    json_path = tmp_path / "empty.json"
    compact_path = tmp_path / "empty.u32le"
    assert chopper.chop("", output="json", dump=json_path) == 0
    assert chopper.chop("", output="compact", dump=compact_path) == 0
    assert json_path.read_text(encoding="utf-8") == "[]"
    assert compact_path.read_bytes() == b""


def test_chop_stream_is_explicitly_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        hiriluk.get_chopper("r50k").chop_stream(iter(["hello"]))


def test_bad_names_and_disabled_model() -> None:
    with pytest.raises(ValueError, match="unknown encoding"):
        hiriluk.get_chopper("not-a-tokenizer")
    with pytest.raises(NotImplementedError, match="properization"):
        hiriluk.get_chopper("llama31p")


def test_alias_is_canonicalized() -> None:
    assert hiriluk.get_chopper("r50k_base").name == "r50k"


def test_input_cannot_be_its_own_dump(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text(TEXT, encoding="utf-8")
    with pytest.raises(ValueError, match="must be different"):
        hiriluk.get_chopper("r50k").chop_file(
            input_path,
            output="compact",
            dump=input_path,
        )


@pytest.mark.skipif(not hasattr(os, "link"), reason="hard links unavailable")
def test_hard_link_cannot_bypass_input_output_guard(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "same-inode.txt"
    input_path.write_text(TEXT, encoding="utf-8")
    os.link(input_path, output_path)
    with pytest.raises(ValueError, match="must be different"):
        hiriluk.get_chopper("r50k").chop_file(
            input_path,
            output="json",
            dump=output_path,
        )


def test_invalid_utf8_file_raises_os_error(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.txt"
    input_path.write_bytes(b"valid prefix\xffinvalid")
    with pytest.raises(OSError, match="invalid UTF-8"):
        hiriluk.get_chopper("r50k").chop_file(
            input_path,
            output="array",
        )


def test_profile_clears_and_tokenizer_resets_after_tokenization_error(
    tmp_path: Path,
) -> None:
    bad_path = tmp_path / "invalid.txt"
    valid_path = tmp_path / "valid.txt"
    output_path = tmp_path / "tokens.u32le"
    suffix = b"<|endof"
    bad_path.write_bytes(b"a" * (256 * 1024 - len(suffix)) + suffix + b"\xff")
    valid_path.write_text("text|>", encoding="utf-8")

    chopper = hiriluk.get_chopper(
        "r50k",
        gigatoken=True,
        profile=True,
    )
    initial = chopper.chop_file(valid_path, output="array")
    assert isinstance(initial, np.ndarray)
    assert initial.dtype == np.uint32
    assert chopper.last_profile is not None

    with pytest.raises(OSError, match="invalid UTF-8"):
        chopper.chop_file(bad_path, output="array")
    assert chopper.last_profile is None

    expected = reference_ids("text|>")
    assert chopper.chop_file(
        valid_path,
        output="compact",
        dump=output_path,
    ) == len(expected)
    profile = chopper.last_profile
    assert isinstance(profile, hiriluk.ChopProfile)
    assert profile.total_tokens == len(expected)
    assert read_u32le(output_path) == expected


def test_dump_io_failures_are_python_os_errors(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        hiriluk.get_chopper("r50k").chop(
            TEXT,
            output="json",
            dump=tmp_path,
        )
