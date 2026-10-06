from collections.abc import Iterable
from os import PathLike
from typing import Literal, NoReturn, TypeAlias, overload

import numpy as np
import numpy.typing as npt

StrPath: TypeAlias = str | PathLike[str]

class ChopProfile:
    @property
    def total_tokens(self) -> int: ...
    @property
    def ttft_seconds(self) -> float | None: ...
    @property
    def elapsed_seconds(self) -> float: ...

class Chopper:
    @property
    def name(self) -> str: ...

    @property
    def gigatoken(self) -> bool: ...

    @property
    def profile(self) -> bool: ...

    @property
    def last_profile(self) -> ChopProfile | None: ...

    @overload
    def chop(
        self,
        text: str,
        *,
        gigatoken: bool | None = None,
        output: Literal["array"] | None = None,
        dump: None = None,
    ) -> npt.NDArray[np.uint32]: ...

    @overload
    def chop(
        self,
        text: str,
        *,
        gigatoken: bool | None = None,
        output: Literal["json", "compact"],
        dump: StrPath,
    ) -> int: ...

    @overload
    def chop(
        self,
        text: str,
        *,
        gigatoken: bool | None = None,
        output: Literal["iterator"],
        dump: None = None,
    ) -> NoReturn: ...

    @overload
    def chop_file(
        self,
        file_path: StrPath,
        *,
        gigatoken: bool | None = None,
        output: Literal["array"] | None = None,
        dump: None = None,
    ) -> npt.NDArray[np.uint32]: ...

    @overload
    def chop_file(
        self,
        file_path: StrPath,
        *,
        gigatoken: bool | None = None,
        output: Literal["json", "compact"],
        dump: StrPath,
    ) -> int: ...

    @overload
    def chop_file(
        self,
        file_path: StrPath,
        *,
        gigatoken: bool | None = None,
        output: Literal["iterator"],
        dump: None = None,
    ) -> NoReturn: ...

    def chop_stream(
        self,
        iterator: Iterable[str],
        *,
        gigatoken: bool | None = None,
        output: Literal["array", "iterator", "json", "compact"] | None = None,
        dump: StrPath | None = None,
    ) -> NoReturn: ...

def get_chopper(
    encoding_name: str,
    *,
    gigatoken: bool = False,
    profile: bool = False,
) -> Chopper: ...
def list_encoding_names() -> list[str]: ...
__version__: str
