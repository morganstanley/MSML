
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import singledispatch

from pydantic import BaseModel


@singledispatch
def _CODEC_DISPATCHER(model_type: type):
    msg = f"No dispatch registered for {model_type=}."
    raise TypeError(msg)


@dataclass(frozen=True)
class ModelCodec:
    """Class for organizing how BaseModel instances are serialized/deserialized."""

    encode: Callable[[BaseModel], str]
    decode: Callable[[str], BaseModel]
    suffix: str


def get_model_codec(model_type: type[BaseModel]) -> ModelCodec:
    return _CODEC_DISPATCHER.dispatch(model_type)(model_type)


get_model_codec.register = _CODEC_DISPATCHER.register


@get_model_codec.register(BaseModel)
def _(model_type: type[BaseModel]) -> ModelCodec:
    return ModelCodec(
        encode=lambda model: model.model_dump_json(indent=2)+"\n",
        decode=lambda string: model_type.model_validate_json(string),
        suffix=".json",
    )
